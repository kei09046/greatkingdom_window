#pragma once
// source : https://github.com/rlcode/per/tree/master

#include <array>
#include <random>
#include "PolicyValue.h"
#include "mcts.h"
#include "GameManager.h"
#include "randm.h"

class GameData {
public:
	std::array<float, 7 * largeSize> state;
	std::array<float, 7 * largeSize> next_state;
	std::array<float, totSize + 1> mcts_probs;
	float winner; 
	// state / action / reward / next state

	GameData();
	GameData(std::array<float, 7 * largeSize>& _state, std::array<float, totSize + 1>& _mcts_probs, 
		float _winner, std::array<float, 7 * largeSize> &_next_state);
	GameData(std::array<float, 7 * largeSize>&& _state, std::array<float, totSize + 1>&& _mcts_probs, 
		float _winner, std::array<float, 7 * largeSize>&& _next_state);
};

class PackedData {
public:
	int idx;
	float diff;
	GameData* gd;
	PackedData(int idx, float diff, GameData* ptr);
	PackedData();
};

class SumTree {
public:
	const static int capacity = 1 << 14;
	int n_elements = 0;
	SumTree();
	void propagate(int idx, float delta);
	void update(int idx, float prior);
	void emplace_back(float prior, std::array<float, 7 * largeSize>& _state,
		std::array<float, totSize + 1>& _mcts_probs, float _winner, std::array<float, 7 * largeSize>& _next_state);
	void emplace_back(float prior, std::array<float, 7 * largeSize>&& _state,
		std::array<float, totSize + 1>&& _mcts_probs, float _winner, std::array<float, 7 * largeSize>&& _next_state);
	int find(int idx, float s) const;
	float total() const;
	PackedData get(float s) const;

private:
	float tree[capacity * 2 - 1];
	int loc = 0;
	GameData* datas[capacity];
};

class Memory {
public:
	float ep = 0.01f;
	float alpha = 0.6f;
	float beta = 0.4f;
	float beta_increment_per_sampling = 0.001f;
	SumTree sum_tree;

	Memory();
	float get_priority(float delta) const;
	void emplace_back(float delta, std::array<float, 7 * largeSize>& _state,
		std::array<float, totSize + 1>& _mcts_probs, float _winner, std::array<float, 7 * largeSize>& _next_state);
	void emplace_back(float delta, std::array<float, 7 * largeSize>&& _state,
		std::array<float, totSize + 1>&& _mcts_probs, float _winner, std::array<float, 7 * largeSize>&& _next_state);
	std::array<PackedData, batchSize> sample();
	void update(int idx, float delta);

private:
};