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


float TrainPipeline::start_play(array<MCTSPlayer*, 2> player_list, ostream& part_res, bool is_shown, float temp) {
	GameManager game_manager = GameManager();
	int idx = 0;
	int move, res, diff;

	while (true) {
		player_list[idx]->get_action(game_manager, part_res, move, is_shown, temp);
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
				part_res << moves.first << moves.second << " ";
		}

		if (res == -2) {
			tie(res, diff) = game_manager.end_game(); // 흑 승 1.0 백 승 -1.0 무승부 0.0
			if(is_shown) part_res << res << endl;
			return (res + 1) * 0.5f; // 흑 승 1.0 백 승 0.0 무승부 0.5
		}

		if ((res == 1 && !idx) || (res == -1 && idx)) {
			if(is_shown) part_res << 1 << endl;
			return 1.0f;
		}
		else {
			if(is_shown) part_res << -1 << endl;
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
			player.get_action(game_manager, cout, inp, shown, temp);
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

float TrainPipeline::policy_evaluate(const std::string& mod_one, const std::string& mod_two, ostream& total_res, ostream& part_res, bool is_shown,
	bool gpu, float temp, int n_games) {
	PolicyValueNet po(mod_one, gpu);
	PolicyValueNet pt(mod_two, gpu);
	MCTSPlayer* base_player = new MCTSPlayer(&po, c_puct, n_playout, /*is_selfplay=*/false);
	MCTSPlayer* oppo_player = new MCTSPlayer(&pt, c_puct, n_playout, false);
	float win_cnt = 0.0f;

	for (int i = 0; i < n_games; ++i) {
		if (!(i % 2))
			win_cnt += TrainPipeline::start_play({ base_player, oppo_player }, part_res, is_shown, temp);
		else
			win_cnt += 1.0f - TrainPipeline::start_play({ base_player, oppo_player }, part_res, is_shown, temp);
		total_res << win_cnt << "/" << i + 1 << endl;
	}

	delete base_player;
	delete oppo_player;
	total_res << win_cnt / static_cast<float>(n_games) << endl;
	return win_cnt / static_cast<float>(n_games);
}

TrainPipeline::TrainPipeline(const std::string& init_model,
	const std::string& test_model, bool gpu, int cnt) : policy_value_net(init_model, gpu),
	prev_policy(test_model, gpu), mcts_player(&policy_value_net, c_puct, n_playout, true), save_cnt(cnt) {
	next_state_batch = new array<float, 7 * batchSize * largeSize>();
	state_batch = new array<float, 7 * batchSize * largeSize>();
	mcts_probs = new array<float, batchSize* (totSize + 1)>();
	winner_batch = new array<float, batchSize>();
	is_weight = new array<float, batchSize>();
}

void TrainPipeline::start_self_play(MCTSPlayer* player, bool is_shown, float temp, int n_games) {
	GameManager game_manager = GameManager();
	int cnt = 0, min_pass = -1;
	pair<int, array<float, totSize + 1>> move_prob;
	passed.fill(false);
	array<float, 7 * largeSize> state[2];
	float t, result;

	while (true) {
		// 처음 두 수 랜덤. 
		cnt++;
		if (cnt < 5) {
			player->get_random_action(game_manager, move_prob.first, is_shown, temp);
			result = game_manager.make_move(move_prob.first, true);
			game_manager.switch_turn();

			if (result) {
				player->reset_player();
				return;
			}
		}

		else {
			state[0] = game_manager.current_state();
			t = player->get_action(game_manager, move_prob, is_shown, temp);

			if (move_prob.first == totSize) {
				passed[cnt] = true;
				if (min_pass == -1)
					min_pass = cnt;
			}
			result = game_manager.make_move(move_prob.first, true);
			game_manager.switch_turn();
			state[1] = game_manager.current_state();

			if (!result) {
				insert_equi_data(abs(player->mcts.pv->evaluate(state[0]) - t), state[0], move_prob.second, t, state[1]);
			}

			else {
				if (result == -2)
					result = -game_manager.end_game().first * game_manager.get_turn();
				insert_equi_data(abs(result - player->mcts.pv->evaluate(state[0])), state[0], move_prob.second, result, state[1]);
				player->reset_player();

				if (1/*is_shown*/) {
					cout << endl;
					for (auto& m : game_manager.get_seqence())
						cout << m.first << "," << m.second << " ";
					cout << endl;

					cout << "episode length : " << game_manager.get_seqence().size() << " winner : " << -result * game_manager.get_turn() << endl;
				}

				return;
			}
		}
	}
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
		(*is_weight)[i] = train_data[i].diff;

		for (int j = 0; j < 7 * largeSize; ++j) {
			(*next_state_batch)[x] = train_data[i].gd->next_state[j];
			(*state_batch)[x++] = train_data[i].gd->state[j];
		}
		for (int j = 0; j < totSize + 1; ++j)
			(*mcts_probs)[y++] = train_data[i].gd->mcts_probs[j];

		(*winner_batch)[i] = train_data[i].gd->winner;
	}

	auto old_probs_value = new pair<array<float, batchSize* (totSize + 1)>, array<float, batchSize> >(policy_value_net.policy_value(state_batch));
	auto new_probs_value = new pair<array<float, batchSize* (totSize + 1)>, array<float, batchSize> >();

	for (int i = 0; i < epochs; ++i) {
		policy_value_net.train_step(*state_batch, *mcts_probs, *winner_batch, *is_weight, learning_rate);
	}

	old_probs_value->second = policy_value_net.policy_value(state_batch).second;
	// 이전 state 와 다음 state의 평가는 반대이므로 -가 아닌 + 필요
	new_probs_value->second = policy_value_net.policy_value(next_state_batch).second;

	for (int i = 0; i < batchSize; ++i) {
		if (abs((*winner_batch)[i]) != 1.0f)
			memory.update(train_data[i].idx, abs(old_probs_value->second[i] + new_probs_value->second[i]));
		else
			memory.update(train_data[i].idx, abs(old_probs_value->second[i] - (*winner_batch)[i]));
	}
	return;
}

// 추가 개선가능점 : process type에 따라 몬테카를로 탐색에서 멀티스레드 적용
float TrainPipeline::policy_evaluate(bool is_shown, float temp, int n_games)
{
	MCTSPlayer* current_player = new MCTSPlayer(&policy_value_net, c_puct, n_playout, /*is_selfplay=*/false);
	MCTSPlayer* past_player = new MCTSPlayer(&prev_policy, c_puct, n_playout, false);
	float win_cnt = 0.0f;

	for (int i = 0; i < n_games; ++i) {
		if (!(i % 2))
			win_cnt += TrainPipeline::start_play({ current_player, past_player }, cout, is_shown, temp);
		else
			win_cnt += 1.0f - TrainPipeline::start_play({ past_player, current_player }, cout, is_shown, temp);
		cout << win_cnt << "/" << i + 1 << endl;
	}

	delete current_player;
	delete past_player;
	return win_cnt / static_cast<float>(n_games);
}

void TrainPipeline::run(bool is_shown, float temp)
{
	string model_file;

	for (int i = 0; i < game_batch_num; ++i) {
		start_self_play(&mcts_player, is_shown, temp, 1);

		if (memory.sum_tree.n_elements > batchSize) {
			policy_update();
		}

		if (!((i + 1 + save_cnt) % save_freq)) {
			model_file = "model3bv7" + to_string(i + 1 + save_cnt);
			policy_value_net.save_model(model_file + string(".pt"));
			cout << "saved" << endl;
		}

		//if (!((i + 1 + save_cnt) % check_freq)) {
		//	cout << "current self-play-batch: " << i + save_cnt << endl;
		//	float win_ratio = 0.0f;
		//	for(int i=0; i<5; ++i)
		//		win_ratio += policy_evaluate("single", true, 20);
		//	cout << win_ratio / 5 << endl;

		//	if (win_ratio > 2.75f) {
		//		model_file += string("best.pt");
		//		policy_value_net.save_model(model_file);
		//		prev_policy.load_model(model_file);
		//	}
		//	else {
		//		memory.clear();
		//		prev_policy.save_model("model.pt");
		//		policy_value_net.load_model("model.pt");
		//	}
		//}
	}
}