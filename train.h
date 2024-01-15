#pragma once
#include "mcts.h"
#include "memory.h"
#include "PolicyValue.h"
#include <string>
#include <deque>
#include <utility>
#include <string>
#include <random>
#include <algorithm>
#include <iostream>
#include <fstream>

static const int n_playout = 800;
static const int play_batch_size = 1;
static const int epochs = 10;
static const int check_freq = 2000;
static const int save_freq = 100;
static const int game_batch_num = 15000;
static const int capacity = 10000;
static float c_puct = 3.0f;

class TrainPipeline {
private:
	float learning_rate = 0.01f;
	float lr_multiplier = 1.0f;

	float kl_targ = 0.02f;
	unsigned int save_cnt = 0;
	unsigned int sample_idxs[capacity];
	//array<float, 7 * largeSize> state_buffer[totSize];
	//array<float, totSize + 1> action_buffer[totSize];
	//float reward_buffer[totSize];
	array<bool, totSize + 1> passed;
	Memory memory;
	PolicyValueNet prev_policy;
	PolicyValueNet policy_value_net;
	MCTSPlayer mcts_player;

public:
	std::array<float, 7 * batchSize * largeSize>* state_batch;
	std::array<float, 7 * batchSize * largeSize>* next_state_batch;
	std::array<float, batchSize * (totSize + 1)>* mcts_probs;
	std::array<float, batchSize>* winner_batch;
	std::array<float, batchSize>* is_weight;

	static float start_play(std::array<MCTSPlayer*, 2> player_list, // 서로 다른 모델들끼리 경기(학습확인용)
		ostream& part_res, bool is_shown = false, float temp = 0.1f);
	static void play(const std::string& model, bool color, int playout, float temp, bool gpu, bool shown); // 사람과 경기
	static float policy_evaluate(const std::string& mod_one, const std::string& mod_two, 
		ostream& total_res, ostream& part_res, bool is_shown = false, bool gpu = true, float temp = 1.0f, int n_games = 100);

	TrainPipeline(const std::string& init_model,
		const std::string& test_model, bool gpu = false, int cnt = 0);

	void start_self_play(MCTSPlayer* player, bool is_shown = false, float temp = 0.1f, int n_games = 1); // 학습 중 경기(학습용)
	void insert_equi_data(float delta, std::array<float, 7 * largeSize>& _state,
		std::array<float, totSize + 1>& _mcts_probs, float _winner, std::array<float, 7 * largeSize>& _next_state);
	void policy_update();
	float policy_evaluate(bool is_shown = false, float temp = 1.0f, int n_games = 100);
	void run(const bool is_shown, float temp);
};