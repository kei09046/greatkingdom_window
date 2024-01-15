#pragma once

#include <utility>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <tuple>
#include <functional>
#include "PolicyValue.h"
#include "randm.h"

const static float fpu_reduction = 0.0f;
const static float dir_portion = 0.0f;

class MCTS_node {
private:
	bool leaf = true;

public:
	std::array<MCTS_node*, totSize + 1> children;
	MCTS_node* parent;
	int _n_visits;
	float _Q, _U, _P;

	MCTS_node(MCTS_node* parent, float prior_p);
	MCTS_node(MCTS_node* parent, float prior_p, float parent_q);
	//~MCTS_node();
	void expand(const std::array<float, totSize + 1>& probs, float parent_q);
	float get_value(float c_puct);
	void update(float leaf_value);
	void update_recursive(float leaf_value);
	int select(float c_puct);
	bool is_leaf() const;
	bool is_root() const;
};

class MCTS {
private:
	const float _c_puct;
	const int _n_playout;
	
public:
	MCTS_node* root;
	PolicyValueNet* pv;

	MCTS(PolicyValueNet* net, float c_puct = 3.0f, int n_playout = 10000);
	~MCTS();
	void delete_tree(MCTS_node* base);
	void _playout(GameManager game_manager);
	std::array<float, totSize + 1> get_move_probs(const GameManager& game_manager, ostream& stm, float temp=0.1f, bool is_shown=false);
	void update_with_move(int last_move);
};

class MCTSPlayer {
private:
	const bool _is_selfplay;
	const float alpha = 0.3f;
	bool player;
	array<float, totSize + 1> dirichlet;

	float get_win_prob();

public:
	MCTS mcts;

	MCTSPlayer(PolicyValueNet* net, int c_puct=5, int n_playout=2000, bool is_selfplay=false);
	void set_player_ind(bool p);
	void reset_player();
	float get_action(const GameManager& game_manager, ostream& stm, int& r, bool shown = false, float temp = 0.1f);
	//void get_action(const GameManager& game_manager, std::array<float, totSize + 1>& r, bool shown = false, float temp = 0.1f);
	float get_action(const GameManager& game_manager, std::pair<int, std::array<float, totSize + 1> >& r, bool shown = false, float temp = 0.1f);
	void get_random_action(const GameManager& game_manager, int& r, bool shown = false, float temp = 0.1f);
};