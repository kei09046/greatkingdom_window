#pragma once

#include <utility>
#include <vector>
#include <array>
#include <set>

const int Max = 1 << 16;
const int boardSize = 9;
const int totSize = 81;
const int largeSize = 121;
const float penalty = 2.5f;
const std::pair<int, int> neutral[1] = { {4, 4} };
const std::pair<int, int> adj[4] = { {1, 0}, {-1, 0}, {0, 1}, {0, -1} };

class GameManager {
public:
	static std::pair<int, int> convert(int no);
	static int convert(int x, int y);

	GameManager();
	std::array<float, 7 * largeSize> current_state() const;
	std::pair<int, float> end_game() const;
	int make_move(int x, int y, bool train_ai);
	int make_move(int z, bool train_ai);
	void calc(int where, bool train_ai);
	void switch_turn();
	void display_board();
	const std::vector<int>& get_available() const;
	const std::pair<int, int>& get_territory() const;
	const std::vector<std::pair<int, int> >& get_seqence() const;
	int get_turn() const;
	bool legal(int cord) const;

private:
	static void remove(std::vector<int>& v, int x);
	int turn, st_cnt, area_cnt;
	int pp = 0;
	std::pair<int, int> territory;
	int board[boardSize + 2][boardSize + 2];
	int st_board[boardSize + 2][boardSize + 2];
	// liberty_list : 돌의 활로를 오름차순 정렬하여 저장
	std::set<int> liberty_list[totSize];
	int bound_board[boardSize + 2][boardSize + 2];
	int terr_board[boardSize + 2][boardSize + 2];
	int eq_list[totSize + 2];
	bool cl[2][boardSize + 2][boardSize + 2];
	std::vector<int> available;
	std::vector<std::pair<int, int> > seq;

	void modify(int n, int m);
};


