#include "PolicyValue.h"
#include "train.h"
#include "mcts.h"
#include "memory.h"
#include <string>
#include <deque>
#include <utility>
#include <string>
#include <random>
#include <algorithm>
using namespace std;


void TrainPipeline::start_self_play(MCTSPlayer* player, bool is_shown, float temp, int n_games) {
	GameManager game_manager = GameManager();
	int result, cnt = 1;
	float t;
	pair<int, array<float, totSize + 1>> move_prob;
	array<float, 7 * largeSize> state[2];

	while (true) {
		// 처음 네 수 랜덤. 
		if(cnt < 5) {
			player->get_random_action(game_manager, move_prob.first, is_shown, temp);
			result = game_manager.make_move(move_prob.first, true);
			game_manager.switch_turn();
			cnt++;

			if (result) {
				player->reset_player();
				return;
			}
		}

		else {
			state[0] = game_manager.current_state();
			t = player->get_action(game_manager, move_prob, is_shown, temp);
			result = game_manager.make_move(move_prob.first, true);
			game_manager.switch_turn();

			state[1] = game_manager.current_state();
			if (!result) {
				//t = -player->mcts.pv->evaluate(state[1]);
				insert_equi_data(abs(player->mcts.pv->evaluate(state[0]) - t), state[0], move_prob.second, t, state[1]);
				//memory.emplace_back(abs(Q - player->mcts.pv->evaluate(state)), move(state), move(move_prob.second), Q, game_manager.current_state());
			}

			else {
				cout << endl;
				if (result == -2)
					result = -game_manager.end_game().first * game_manager.get_turn();

				insert_equi_data(abs(result - player->mcts.pv->evaluate(state[0])), state[0], move_prob.second, result, state[1]);
				//memory.emplace_back(abs(result - player->mcts.pv->evaluate(state)), move(state), move(move_prob.second), result, game_manager.current_state());

				player->reset_player();
				for (auto& m : game_manager.get_seqence())
					cout << m.first << "," << m.second << " ";
				cout << endl;

				cout << "episode length : " << game_manager.get_seqence().size() << " winner : " << -result * game_manager.get_turn() << endl;
				return;
			}
		}
	}
}

float TrainPipeline::start_play(array<MCTSPlayer*, 2> player_list, bool is_shown, float temp) {
	GameManager game_manager = GameManager();
	int idx = 0;
	int move, res, diff;

	while (true) {
		player_list[idx]->get_action(game_manager, move, is_shown, temp);
		res = game_manager.make_move(move, true);
		game_manager.switch_turn();

		if (!res) {
			idx = 1 - idx;
			continue;
		}
		else {
			player_list[0]->reset_player();
			player_list[1]->reset_player();
		}

		if (is_shown) {
			auto& seq = game_manager.get_seqence();
			for (auto& moves : seq)
				cout << moves.first << moves.second << " ";
		}

		if (res == -2) {
			tie(res, diff) = game_manager.end_game();
			switch (res) {
			case 0:
				cout << "draw" << endl;
				return 0.5f;
			case -1:
				cout << "second player wins" << endl;
				return 0.0f;
			case 1:
				cout << "first player wins" << endl;
				return 1.0f;
			}
		}

		if ((res == 1 && !idx) || (res == -1 && idx)) {
			cout << "first player wins" << endl;
			return 1.0f;
		}
		else {
			cout << "second player wins" << endl;
			return 0.0f;
		}
	}
}

void TrainPipeline::play(const string& model, bool color, int playout, float temp, bool gpu, bool shown) {
	GameManager game_manager = GameManager();
	PolicyValueNet* pv = new PolicyValueNet(model, gpu);
	MCTSPlayer player = MCTSPlayer(pv, 2, playout, false);
	pair<int, int> cord;
	int inp, res;

	while (true) {
		if (color) {
			cin >> cord.first >> cord.second;
		}
		else {
			player.get_action(game_manager, inp, shown, temp);
			cord.first = inp / boardSize;
			cord.second = inp % boardSize;
		}
		color = !color;
		res = game_manager.make_move(cord.first, cord.second, true);
		game_manager.switch_turn();
		game_manager.display_board();
		if (res) {
			// 게임 종료 화면
			break;
		}
	}
	return;
}

TrainPipeline::TrainPipeline(const std::string& init_model,
	const std::string& test_model, bool gpu, int cnt) : policy_value_net(init_model, gpu),
	prev_policy(test_model, gpu), mcts_player(&policy_value_net, c_puct, n_playout, true), cnt(cnt) {
	next_state_batch = new array<float, 7 * batchSize * largeSize>();
	state_batch = new array<float, 7 * batchSize * largeSize>();
	mcts_probs = new array<float, batchSize* (totSize + 1)>();
	winner_batch = new array<float, batchSize>();
}

void TrainPipeline::insert_equi_data(float delta, std::array<float, 7 * largeSize>& _state,
	std::array<float, totSize + 1>& _mcts_probs, float _winner, std::array<float, 7 * largeSize>& _next_state) {

	std::array<float, 7 * largeSize> temp_state;
	std::array<float, 7 * largeSize> temp_next_state;
	std::array<float, totSize + 1> temp_mcts_probs;
	int cnt, dnt, ind;

	// insert original data
	memory.emplace_back(delta, _state, _mcts_probs, _winner, _next_state);

	// insert rotated data
	cnt = 0;
	dnt = 0;	 
	for (int j = 0; j < 7; ++j)
		for (int k = 0; k < boardSize + 2; ++k)
			for (int l = 0; l < boardSize + 2; ++l) {
				ind = j * largeSize + l * (boardSize + 2) + k;
				temp_next_state[cnt] = _next_state[ind];
				temp_state[cnt++] = _state[ind];
			}

	for (int k = 0; k < boardSize; ++k)
		for (int l = 0; l < boardSize; ++l)
			temp_mcts_probs[dnt++] = _mcts_probs[l * boardSize + k];

	temp_mcts_probs[totSize] = _mcts_probs[totSize];
	memory.emplace_back(delta, move(temp_state), move(temp_mcts_probs), _winner, move(temp_next_state));


	cnt = 0;
	dnt = 0;
	for (int j = 0; j < 7; ++j)
		for (int k = 0; k < boardSize + 2; ++k)
			for (int l = 0; l < boardSize + 2; ++l) {
				ind = j * largeSize + (k + 1) * (boardSize + 2) - (l + 1);
				temp_next_state[cnt] = _next_state[ind];
				temp_state[cnt++] = _state[ind];
			}

	for (int k = 0; k < boardSize; ++k)
		for (int l = 0; l < boardSize; ++l)
			temp_mcts_probs[dnt++] = _mcts_probs[(k + 1) * boardSize - (l + 1)];

	temp_mcts_probs[totSize] = _mcts_probs[totSize];
	memory.emplace_back(delta, move(temp_state), move(temp_mcts_probs), _winner, move(temp_next_state));


	cnt = 0;
	dnt = 0;
	for (int j = 0; j < 7; ++j)
		for (int k = 0; k < boardSize + 2; ++k)
			for (int l = 0; l < boardSize + 2; ++l) {
				ind = j * largeSize + (k + 1) * (boardSize + 2) - (l + 1);
				temp_next_state[cnt] = _next_state[ind];
				temp_state[cnt++] = _state[ind];
			}

	for (int k = 0; k < boardSize; ++k)
		for (int l = 0; l < boardSize; ++l)
			temp_mcts_probs[dnt++] = _mcts_probs[(l + 1) * boardSize - (k + 1)];

	temp_mcts_probs[totSize] = _mcts_probs[totSize];
	memory.emplace_back(delta, move(temp_state), move(temp_mcts_probs), _winner, move(temp_next_state));


	cnt = 0;
	dnt = 0;
	for (int j = 0; j < 7; ++j)
		for (int k = 0; k < boardSize + 2; ++k)
			for (int l = 0; l < boardSize + 2; ++l) {
				ind = j * largeSize + (boardSize + 1 - k) * (boardSize + 2) + l;
				temp_next_state[cnt] = _next_state[ind];
				temp_state[cnt++] = _state[ind];
			}

	for (int k = 0; k < boardSize; ++k)
		for (int l = 0; l < boardSize; ++l)
			temp_mcts_probs[dnt++] = _mcts_probs[(boardSize - 1 - k) * boardSize + l];

	temp_mcts_probs[totSize] = _mcts_probs[totSize];
	memory.emplace_back(delta, move(temp_state), move(temp_mcts_probs), _winner, move(temp_next_state));


	cnt = 0;
	dnt = 0;
	for (int j = 0; j < 7; ++j)
		for (int k = 0; k < boardSize + 2; ++k)
			for (int l = 0; l < boardSize + 2; ++l) {
				ind = j * largeSize + (boardSize + 1 - l) * (boardSize + 2) + k;
				temp_next_state[cnt] = _next_state[ind];
				temp_state[cnt++] = _state[ind];
			}

	for (int k = 0; k < boardSize; ++k)
		for (int l = 0; l < boardSize; ++l)
			temp_mcts_probs[dnt++] = _mcts_probs[(boardSize - 1 - l) * boardSize + k];

	temp_mcts_probs[totSize] = _mcts_probs[totSize];
	memory.emplace_back(delta, move(temp_state), move(temp_mcts_probs), _winner, move(temp_next_state));


	cnt = 0;
	dnt = 0;
	for (int j = 0; j < 7; ++j)
		for (int k = 0; k < boardSize + 2; ++k)
			for (int l = 0; l < boardSize + 2; ++l) {
				ind = j * largeSize + (boardSize + 2 - k) * (boardSize + 2) - (l + 1);
				temp_next_state[cnt] = _next_state[ind];
				temp_state[cnt++] = _state[ind];
			}

	for (int k = 0; k < boardSize; ++k)
		for (int l = 0; l < boardSize; ++l)
			temp_mcts_probs[dnt++] = _mcts_probs[(boardSize - k) * boardSize - (l + 1)];

	temp_mcts_probs[totSize] = _mcts_probs[totSize];
	memory.emplace_back(delta, move(temp_state), move(temp_mcts_probs), _winner, move(temp_next_state));


	cnt = 0;
	dnt = 0;
	for (int j = 0; j < 7; ++j)
		for (int k = 0; k < boardSize + 2; ++k)
			for (int l = 0; l < boardSize + 2; ++l) {
				ind = j * largeSize + (boardSize + 2 - l) * (boardSize + 2) - (k + 1);
				temp_next_state[cnt] = _next_state[ind];
				temp_state[cnt++] = _state[ind];
			}

	for (int k = 0; k < boardSize; ++k)
		for (int l = 0; l < boardSize; ++l)
			temp_mcts_probs[dnt++] = _mcts_probs[(boardSize - l) * boardSize - (k + 1)];

	temp_mcts_probs[totSize] = _mcts_probs[totSize];
	memory.emplace_back(delta, move(temp_state), move(temp_mcts_probs), _winner, move(temp_next_state));

	return;
}

void TrainPipeline::policy_update()
{
	array<PackedData, batchSize> train_data = memory.sample();
	int x = 0, y = 0;
	for (int i = 0; i < batchSize; ++i) {
		//is_weight[i] = train_data[i].diff;

		for (int j = 0; j < 7 * largeSize; ++j) {
			(*next_state_batch)[x] = train_data[i].gd->next_state[j];
			(*state_batch)[x++] = train_data[i].gd->state[j];
		}
		for (int j = 0; j < totSize + 1; ++j)
			(*mcts_probs)[y++] = train_data[i].gd->mcts_probs[j];

		(*winner_batch)[i] = train_data[i].gd->winner;
		//cout << winner_batch[i] << endl;
	}

	auto old_probs_value = new pair<array<float, batchSize* (totSize + 1)>, array<float, batchSize> >(policy_value_net.policy_value(state_batch));
	auto new_probs_value = new pair<array<float, batchSize* (totSize + 1)>, array<float, batchSize> >();

	float ov, nv;
	float kl = 0.0f;
	for (int i = 0; i < epochs; ++i) {
		kl = 0.0f;
		policy_value_net.train_step(*state_batch, *mcts_probs, *winner_batch, is_weight, learning_rate * lr_multiplier);
		*new_probs_value = policy_value_net.policy_value(state_batch);
		for (int j = 0; j < batchSize * (totSize + 1); ++j) {
			ov = old_probs_value->first[j];
			nv = new_probs_value->first[j];
			kl += ov * (log(ov + 0.0001f) - log(nv + 0.0001f));
		}
		kl /= batchSize;
		if (kl > kl_targ * 4)
			break;
	}

	if (kl > kl_targ * 2 && lr_multiplier > 0.1f)
		lr_multiplier /= 1.5f;
	else if (kl < kl_targ / 2 && lr_multiplier < 10.0f)
		lr_multiplier *= 1.5f;

	old_probs_value->second = policy_value_net.policy_value(state_batch).second;
	// 이전 state 와 다음 state의 평가는 반대이므로 -가 아닌 + 필요
	new_probs_value->second = policy_value_net.policy_value(next_state_batch).second;

	for (int i = 0; i < batchSize; ++i) {
		if(abs((*winner_batch)[i]) != 1.0f)
			memory.update(train_data[i].idx, abs(old_probs_value->second[i] + new_probs_value->second[i]));
		else
			memory.update(train_data[i].idx, abs(old_probs_value->second[i] - (*winner_batch)[i]));
	}

	delete old_probs_value;
	delete new_probs_value;
	return;
}

// 추가 개선가능점 : process type에 따라 몬테카를로 탐색에서 멀티스레드 적용
float TrainPipeline::policy_evaluate(const std::string& process_type, bool is_shown, int n_games)
{
	MCTSPlayer* current_player = new MCTSPlayer(&policy_value_net, c_puct, n_playout, /*is_selfplay=*/false);
	MCTSPlayer* past_player = new MCTSPlayer(&prev_policy, c_puct, n_playout, false);
	float win_cnt = 0.0f;

	for (int i = 0; i < n_games; ++i) {
		if (!(i % 2))
			win_cnt += TrainPipeline::start_play({ current_player, past_player }, is_shown, temp);
		else
			win_cnt += 1.0f - TrainPipeline::start_play({ past_player, current_player }, is_shown, temp);
		cout << win_cnt << "/" << i + 1 << endl;
	}

	delete current_player;
	delete past_player;
	return win_cnt / static_cast<float>(n_games);
}

void TrainPipeline::run()
{
	string model_file;

	for (int i = 0; i < game_batch_num; ++i) {
		start_self_play(&mcts_player, true, temp, 1);

		if (memory.sum_tree.n_elements > batchSize) {
			policy_update();
		}

		if (!((i + 1 + cnt) % save_freq)) {
			model_file = "model3bv5";
			model_file += to_string(i + 1 + cnt);
			policy_value_net.save_model(model_file + string(".pt"));
			cout << "saved" << endl;
		}

		if (!((i + 1 + cnt) % check_freq)) {
			cout << "current self-play-batch: " << i + cnt << endl;
			float win_ratio = 0.0f;
			for(int i=0; i<5; ++i)
				win_ratio += policy_evaluate("single", true, 20);
			cout << win_ratio / 5 << endl;

			if (win_ratio > 2.75f) {
				model_file += string("best.pt");
				policy_value_net.save_model(model_file);
				prev_policy.load_model(model_file);
			}
			else {
				prev_policy.save_model("model.pt");
				policy_value_net.load_model("model.pt");
				memory.beta -= memory.beta_increment_per_sampling * check_freq;
			}
		}
	}
}