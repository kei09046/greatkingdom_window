#pragma once
#include <string>
#include "mcts.h"
#include "PolicyValue.h"
#include "memory.h"
#include <deque>
#include <utility>
#include <string>
#include <random>
#include <algorithm>

class TrainPipeline {
private:
	const int n_playout = 800;
	const int play_batch_size = 1;
	const int epochs = 10;
	const int check_freq = 1000;
	const int save_freq = 50;
	const int game_batch_num = 5000;
	float learning_rate = 0.003f;
	float lr_multiplier = 1.0f;
	float temp = 1.0f;
	float c_puct = 3.0f;

	Memory memory;
	float kl_targ = 0.02f;
	int cnt = 0;
	int episode_len = 0;
	PolicyValueNet prev_policy;
	PolicyValueNet policy_value_net;
	MCTSPlayer mcts_player;

public:
	std::array<float, 7 * batchSize * largeSize>* state_batch;
	std::array<float, 7 * batchSize * largeSize>* next_state_batch;
	std::array<float, batchSize* (totSize + 1)>* mcts_probs;
	std::array<float, batchSize>* winner_batch;
	std::array<float, batchSize> is_weight;

	static float start_play(std::array<MCTSPlayer*, 2> player_list,
		bool is_shown = false, float temp = 0.1f);
	static void play(const std::string& model, bool color, int playout, float temp, bool gpu, bool shown);

	TrainPipeline(const std::string& init_model,
		const std::string& test_model, bool gpu = false, int cnt = 0);

	void start_self_play(MCTSPlayer* player, bool is_shown = false, float temp = 0.1f, int n_games = 1);
	void insert_equi_data(float delta, std::array<float, 7 * largeSize>& _state,
		std::array<float, totSize + 1>& _mcts_probs, float _winner, std::array<float, 7 * largeSize>& _next_state);
	void policy_update();
	float policy_evaluate(const std::string& process_type, bool is_shown = false, int n_games = 100);
	void run();
};