#include "GameManager.h"
#include <iostream>
#include <algorithm>
using namespace std;


pair<int, int> GameManager::convert(int no) {
	return { no / boardSize, no % boardSize };
}

int GameManager::convert(int x, int y) {
	return x * boardSize + y;
}

GameManager::GameManager() {
	this->turn = 1;
	this->pp = 0;
	this->territory = { 0, 0 };
	this->area_cnt = 0;
	this->st_cnt = 0;
	for (int i = 0; i < boardSize + 2; ++i) {
		for (int j = 0; j < boardSize + 2; ++j) {
			if (i == 0 || i == boardSize + 1 || j == 0 || j == boardSize + 1) {
				this->board[i][j] = Max;
				this->st_board[i][j] = Max;
				this->bound_board[i][j] = -1;
			}
			else {
				this->board[i][j] = 0;
				this->st_board[i][j] = 0;
				this->bound_board[i][j] = 0;
			}
			this->terr_board[i][j] = 0;
			this->cl[0][i][j] = false;
			this->cl[1][i][j] = false;
		}
	}

	for (int i = 0; i < totSize; ++i) {
		this->liberty_list[i] = {};
		this->available.push_back(i);
	}
	// pass 추가
	this->available.push_back(totSize);

	for (const pair<int, int>& i : neutral) {
		this->remove(this->available, this->convert(i.first, i.second));
		this->board[i.first + 1][i.second + 1] = Max;
		this->st_board[i.first + 1][i.second + 1] = Max;
		this->bound_board[i.first + 1][i.second + 1] = -1;
	}
}

// python 버전과는 달리 일차원 반환 크기도 다름.(중립돌 + 경계를 별도 레이어로 전환)

array<float, 7 * largeSize> GameManager::current_state() const{
	array<float, 7 * largeSize> ret;
	ret.fill(0.0f);
	
	int cnt = 0;
	float c = static_cast<float>((territory.first - territory.second - penalty) * turn);
	for(int i= 0; i < boardSize + 2; ++i)
		for (int j = 0; j < boardSize + 2; ++j) {
			switch (turn * board[i][j]) {
			case 1:
				ret[cnt] = 1.0f;
				break;
			case -1:
				ret[largeSize + cnt] = 1.0f;
				break;
			case Max:
				ret[2 * largeSize + cnt] = 1.0f;
				break;
			case -Max:
				ret[2 * largeSize + cnt] = 1.0f;
				break;
			default:
				break;
			}

			ret[5 * largeSize + cnt] = static_cast<float>(turn * terr_board[i][j]);
			ret[4 * largeSize + cnt] = static_cast<float>(turn);
			ret[6 * largeSize + cnt] = c;
			cnt++;
		}

	//cout << turn << endl;
	if (!this->seq.empty()) {
		const pair<int, int> last = this->seq.back();
		ret[3 * largeSize + (last.first + 1) * (boardSize + 2) + last.second + 1] = 1.0f;
	}

	return ret;
}

pair<int, float> GameManager::end_game() const {
	int b_score = this->territory.first;
	int w_score = this->territory.second;
	float diff = b_score - w_score - penalty;
	if (diff > 0)
		return { 1, diff };
	if (diff == 0)
		return { 0, 0 };
	return { -1, -diff };
}

int GameManager::make_move(int x, int y, bool train_ai)
{
	seq.push_back({ x, y });
	int z = convert(x, y);

	if (x >= boardSize && pp == 1) {
		return -2;
	}
	if (x >= boardSize) {
		pp++;
		return 0;
	}

	if (train_ai) {
		remove(available, z);
	}

	x++; y++;
	//돌이 있는 곳에 착수했는지 체크
	if (st_board[x][y])
		return 0;
	// 상대 영역에 착수했는지 체크
	if (terr_board[x][y] * turn == -1)
		return -1;

	pp = 0;

	//활로 계산, 잡힌 돌 있는지 판단
	board[x][y] = turn;
	//cout << x << " " << y << endl;
	set<int> empty_space;
	int min = Max;
	int s = 0;
	int same_adj[4] = { -1, -1, -1, -1 };
	int diff_adj[4] = { -1, -1, -1, -1 };

	for (int i = 0; i < 4; ++i) {
		int u = this->board[x + adj[i].first][y + adj[i].second];
		if (!u)
			empty_space.insert(convert(x + adj[i].first - 1, y + adj[i].second - 1));
		else if (u == Max)
			continue;
		else if (u * turn > 0) {
			int v = this->st_board[x + adj[i].first][y + adj[i].second];
			if (min > v)
				min = v;
			for (int j = 0; j < 4; ++j) {
				if (same_adj[j] == -1) {
					same_adj[j] = v;
					s++;
					break;
				}
				if (same_adj[j] == v)
					break;
			}

		}
		else {
			int v = this->st_board[x + adj[i].first][y + adj[i].second];
			for (int j = 0; j < 4; ++j) {
				if (diff_adj[j] == -1) {
					diff_adj[j] = v;
					break;
				}
				if (diff_adj[j] == v)
					break;
			}

		}
	}

	//인접한 상대 돌의 활로 감소
	for (int i = 0; i < 4; ++i) {
		if (diff_adj[i] == -1)
			break;
		this->liberty_list[diff_adj[i]].erase(z);
		if (liberty_list[diff_adj[i]].empty())
			return 1;
	}

	// 인접한 내 돌이 없는 경우
	if (!s) {
		this->st_board[x][y] = ++st_cnt;
		this->liberty_list[st_cnt] = move(empty_space);
		min = st_cnt;
	}

	//인접한 내 돌이 있는 경우
	else {
		this->st_board[x][y] = min;
		for (int k : same_adj) {
			if (k == -1)
				break;
			if (k == min)
				continue;

			liberty_list[min].insert(liberty_list[k].begin(), liberty_list[k].end());
			//this->liberty_list[k] = 0;

			for (int i = 1; i < boardSize + 1; ++i)
				for (int j = 1; j < boardSize + 1; ++j)
					if (this->st_board[i][j] == k)
						this->st_board[i][j] = min;
		}

		liberty_list[min].insert(empty_space.begin(), empty_space.end());
		liberty_list[min].erase(z);
		//this->liberty_list[min] += empty_space - s;
	}

	if (liberty_list[min].empty())
		return -1;

	// 영역 갱신
	this->bound_board[x][y] = -1;
	int temp = (1 - this->turn) / 2;

	for (int i = 0; i < 4; ++i)
		this->cl[temp][x + adj[i].first][y + adj[i].second] = true;

	//자기 집 안에 착수
	if (this->terr_board[x][y] == this->turn) {
		if(train_ai)
			std::cout << "error : placed in self territory" << std::endl;
		if (temp)
			this->territory.second--;
		else
			this->territory.first--;
		this->terr_board[x][y] = 0;
	}

	// 공배에 착수
	else {
		for (int i = 0; i < 4; ++i) {
			if(this->bound_board[x + adj[i].first][y + adj[i].second] != -1){
				this->calc(this->bound_board[x + adj[i].first][y + adj[i].second], train_ai);
				break;
			}
		}
	}

	return 0;
}

int GameManager::make_move(int z, bool train_ai) {
	return make_move(convert(z).first, convert(z).second, train_ai);
}

void GameManager::calc(int where, bool train_ai)
{
	if (where < 0)
		return;
	
	int c[boardSize + 2][boardSize + 2];
	int cnt = 0;
	bool adj[4][6];

	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < 6; ++j)
			adj[i][j] = false;

	int N = totSize + 2;
	for (int i = 0; i < N; ++i)
		eq_list[i] = i;

	for(int i=1; i<boardSize + 1; ++i)
		for (int j = 1; j < boardSize + 1; ++j)
			if (bound_board[i][j] == where) {
				if (bound_board[i - 1][j] == -1 && bound_board[i][j - 1] == -1)
					c[i][j] = cnt++;
				else if (bound_board[i - 1][j] == -1)
					c[i][j] = c[i][j - 1];
				else if (bound_board[i][j - 1] == -1)
					c[i][j] = c[i - 1][j];
				else if (c[i - 1][j] == c[i][j - 1])
					c[i][j] = c[i - 1][j];
				else if (c[i - 1][j] < c[i][j - 1]) {
					c[i][j] = c[i - 1][j];
					modify(c[i][j - 1], c[i - 1][j]);
				}
				else {
					c[i][j] = c[i][j - 1];
					modify(c[i - 1][j], c[i][j - 1]);
				}
			}

	int dnt = 0;
	for (int i = 0; i < cnt; ++i) {
		if (i == eq_list[i])
			this->eq_list[i] = dnt++;
		else
			this->eq_list[i] = this->eq_list[this->eq_list[i]];
	}

	for(int i=1; i <= boardSize; ++i)
		for (int j = 1; j <= boardSize; ++j) {
			if (bound_board[i][j] > where)
				bound_board[i][j]--;
			else if (bound_board[i][j] == where) {
				bound_board[i][j] = area_cnt + eq_list[c[i][j]];
				int d = eq_list[c[i][j]];
				if (i == 1)
					adj[d][0] = true;
				if (j == 1)
					adj[d][1] = true;
				if (i == boardSize)
					adj[d][2] = true;
				if (j == boardSize)
					adj[d][3] = true;
				if (this->cl[0][i][j])
					adj[d][4] = true;
				if (this->cl[1][i][j])
					adj[d][5] = true;
			}

		}

	bool t[4][2];
	for (int i = 0; i < dnt; ++i) {
		t[i][0] = !((adj[i][0] && adj[i][1] && adj[i][2] && adj[i][3]) || adj[i][5]);
		t[i][1] = !((adj[i][0] && adj[i][1] && adj[i][2] && adj[i][3]) || adj[i][4]);
	}

	for(int i=1; i<boardSize + 1; ++i)
		for (int j = 1; j < boardSize + 1; ++j) {
			int u = this->bound_board[i][j] - this->area_cnt;
			if (u >= 0) {
				if (t[u][0]) {
					/*std::cout << "black territory : " << i-1 << " " << j-1 << std::endl;*/
					this->terr_board[i][j] = 1;
					this->territory.first++;
					if (train_ai)
						this->remove(this->available, this->convert(i - 1, j - 1));
				}
				else if (t[u][1]) {
					/*std::cout << "white territory changed" << std::endl;*/
					this->terr_board[i][j] = -1;
					this->territory.second++;
					if (train_ai)
						this->remove(this->available, this->convert(i - 1, j - 1));
				}
			}
		}

	//for (auto i : available)
	//	std::cout << i << " ";
	//std::cout << std::endl;
	this->area_cnt += dnt - 1;
	return;
}

void GameManager::switch_turn() {
	turn *= -1;
	return;
}

void GameManager::display_board() {
	for (int i = 1; i <= boardSize; ++i) {
		for (int j = 1; j <= boardSize; ++j) {
			switch (board[i][j]) {
			case 0:
				cout << '-';
				break;
			case -1:
				cout << 'o';
				break;
			case 1:
				cout << 'x';
				break;
			default:
				cout << '+';
				break;
			}
		}
		cout << endl;
	}

	cout << endl << endl;
	return;
}

void GameManager::modify(int n, int m)
{
	int eqv = this->eq_list[n];
	if (eqv == n) {
		this->eq_list[n] = m;
		return;
	}
	if (eqv > n) {
		this->eq_list[n] = m;
		modify(eqv, m);
		return;
	}
	this->eq_list[n] = min(eqv, m);
	modify(max(eqv, m), min(eqv, m));
	return;
}

void GameManager::remove(vector<int>& v, int x) {
	int N = v.size();
	for (int i = 0; i < N; ++i)
		if (v[i] == x) {
			v.erase(next(v.begin(), i));
			break;
		}
	return;
}

const vector<int>& GameManager::get_available() const
{
	return this->available;
}

const std::pair<int, int>& GameManager::get_territory() const
{
	return this->territory;
}

int GameManager::get_turn() const {
	return this->turn;
}

const vector<pair<int, int> >& GameManager::get_seqence() const{
	return this->seq;
}

bool GameManager::legal(int cord) const {
	int x = cord / boardSize + 1;
	int y = cord % boardSize + 1;
	return (x >= boardSize + 1) || !(board[x][y] || terr_board[x][y]);
}