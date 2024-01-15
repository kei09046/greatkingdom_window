#include "mcts.h"
#include "PolicyValue.h"
using namespace std;
// mctsplayer 의 copy를 생성할 경우 소멸 과정에서 에러가 발생할 수 있음. 코드에서 copy가 생성되지 않도록 할 것


MCTS_node::MCTS_node(MCTS_node* parent, float prior_p) : _P(prior_p), parent(parent),
_n_visits(0), _Q(0.0f), _U(0.0f) {}

MCTS_node::MCTS_node(MCTS_node* parent, float prior_p, float parent_q): _P(prior_p), parent(parent),
_n_visits(0), _Q(min(-parent_q + fpu_reduction, 1.0f)), _U(0.0f) {}

void MCTS_node::expand(const array<float, totSize + 1>& prob, float parent_q)
{
	leaf = false;
	for (int i = 0; i <= totSize; ++i) {
		if (prob[i] != 2.0f) {
			//children[i] = new MCTS_node(this, prob[i], parent_q);
			children[i] = new MCTS_node(this, prob[i]);
		}
		else
			children[i] = nullptr;
	}

	return;
}

float MCTS_node::get_value(float c_puct)
{
	_U = (c_puct * _P * sqrt(parent->_n_visits)) / (1 + _n_visits);
	return _Q + _U;
}

void MCTS_node::update(float leaf_value)
{
	if (!_n_visits)
		_Q = 0.0f;

	_Q += (leaf_value - _Q) / ++_n_visits;
	return;
}

void MCTS_node::update_recursive(float leaf_value)
{
	if (parent != nullptr)
		parent->update_recursive(-leaf_value);
	update(leaf_value);
	return;
}

int MCTS_node::select(float c_puct)
{
	float val = -10.0f;
	float temp;
	int loc = 0;

	for (int i = 0; i <= totSize; ++i) {
		if (children[i] != nullptr) {
			temp = children[i]->get_value(c_puct);
			if (temp > val) {
				val = temp;
				loc = i;
			}
		}
	}

	return loc;
}

bool MCTS_node::is_leaf() const
{
	return leaf; // this 가 nullptr 였다?
}

bool MCTS_node::is_root() const
{
	return (parent == nullptr);
}


MCTS::MCTS(PolicyValueNet* net, float c_puct, int n_playout):
_c_puct(c_puct), _n_playout(n_playout), pv(net){
	root = new MCTS_node(nullptr, 1.0f);
}

MCTS::~MCTS() {
	delete_tree(root);
}

void MCTS::delete_tree(MCTS_node* base) {
	if (base == nullptr)
		return;

	if (!base->is_leaf())
		for (int i = 0; i <= totSize; ++i)
			delete_tree(base->children[i]);

	delete base;
}

void MCTS::_playout(GameManager game_manager)
{
	int result = 0;
	int action = 0;
	MCTS_node* node = root;
	array<float, totSize + 1> action_probs;
	float eval;

	
	while (true) {
		if (node->is_leaf())
			break;
		action = node->select(_c_puct);
		
		node = node->children[action];
		//cout << "make move";
		result = game_manager.make_move(action, true);
		//cout << "move";
		game_manager.switch_turn();
	}

	switch (result) {
	case 0:
		tie(action_probs, eval) = pv->policy_value_fn(game_manager);
		node->expand(action_probs, eval);
		//if (eval != 1.0f)
		//	cout << eval;
		eval = -eval;
		break;
	case -2:
		eval = -static_cast<float>(game_manager.end_game().first) * game_manager.get_turn();
		break;
	default:
		eval = static_cast<float>(result);
	}

	node->update_recursive(eval);
}

array<float, totSize + 1> MCTS::get_move_probs(const GameManager& game_manager, ostream& stm, float temp, bool is_shown)
{
	for (int i = 0; i < _n_playout; ++i) {
		//cout << i << endl;
		_playout(game_manager);
	}

	if (is_shown) {
		pair<int, pair<int, float> > M_nodes[3];
		for (int i = 0; i < 3; ++i)
			M_nodes[i].second.first = -1;
		int n;

		for (int i = 0; i < root->children.size(); ++i) {
			if (root->children[i] != nullptr) {
				//stm << (root->children[i])->_n_visits << " " << root->children[i]->_Q << " ";

				n = root->children[i]->_n_visits;
				if (n < M_nodes[2].second.first)
					continue;
				else if (n < M_nodes[1].second.first) {
					M_nodes[2] = { i, {n, root->children[i]->_Q} };
				}
				else if (n < M_nodes[0].second.first) {
					M_nodes[2] = M_nodes[1];
					M_nodes[1] = { i, {n, root->children[i]->_Q} };
				}
				else {
					M_nodes[2] = M_nodes[1];
					M_nodes[1] = M_nodes[0];
					M_nodes[0] = { i, {n, root->children[i]->_Q} };
				}
			}
			//else
			//	cout << 0 << " " << 0 << " ";
		}

		for (int i = 0; i < 3; ++i) {
			stm << i << "th best move : " << M_nodes[i].first / boardSize << " " << M_nodes[i].first % boardSize << " visit count : " <<
				M_nodes[i].second.first << " eval : " << M_nodes[i].second.second << endl;
		}
	}


	array<float, totSize + 1> ret;
	if (temp > 0.3f) {
		float sum = 0.0f;
		for (int i = 0; i <= totSize; ++i) {
			if (root->children[i] == nullptr)
				ret[i] = 0.0f;
			else if (!root->children[i]->_n_visits)
				ret[i] = 0.0f;
			else {
				ret[i] = pow(root->children[i]->_n_visits, 1.0f / temp);
				sum += ret[i];
			}
		}

		for (int i = 0; i <= totSize; ++i)
			ret[i] /= sum;
	}

	// 논문에서 평가대국 시 temp -> 0 계산 시 발산하므로 visit count 가 가장 높은 노드 선택
	else {
		int max = 0, cnt = 0, ind = 0;
		ret.fill(0.0f);
		for (auto r : root->children) {
			if(r != nullptr)
				if (r->_n_visits > max) {
					max = r->_n_visits;
					ind = cnt;
				}

			cnt++;
		}

		ret[ind] = 1.0f;
	}
	return ret;
}

void MCTS::update_with_move(int last_move)
{
	if (last_move >= 0) {
		MCTS_node* temp = root;
		root = root->children[last_move];
		root->parent = nullptr;
		temp->children[last_move] = nullptr;
		delete_tree(temp);
	}

	else {
		delete_tree(root);
		root = new MCTS_node(nullptr, 1.0f);
	}
	/*cout << "update with move" << endl;*/
}

MCTSPlayer::MCTSPlayer(PolicyValueNet* net, int c_puct, int n_playout, bool is_selfplay):
mcts(net, c_puct, n_playout), _is_selfplay(is_selfplay), player(true){
	dirichlet.fill(0.0f);
}


void MCTSPlayer::set_player_ind(bool p)
{
	player = p;
	return;
}

void MCTSPlayer::reset_player()
{
	mcts.update_with_move(-10);
}

float MCTSPlayer::get_action(const GameManager& game_manager, ostream& stm, int& r, bool shown, float temp)
{
	array<float, totSize + 1> move_probs = mcts.get_move_probs(game_manager, stm, temp, shown);
	int move = totSize;
	float rnd = get_random(0.0f, 1.0f);
	float cnt = 0.0f;
	float ret = get_win_prob();

	if (_is_selfplay) {
		// get dirichlet distribution
		int n_legal = game_manager.get_n_available();
		float sum = 0.0f;

		for (int i = 0; i < n_legal; ++i) {
			dirichlet[i] = get_gamma();
			sum += dirichlet[i];
		}
		for (int i = 0; i < n_legal; ++i)
			dirichlet[i] /= sum;

		int dnt = 0;
		for (int i : game_manager.get_available()) {
			cnt += move_probs[i] * (1.0f - dir_portion) + dirichlet[dnt++] * dir_portion;
			if (cnt > rnd) {
				mcts.update_with_move(i);
				move = i;
				break;
			}
		}
	}

	else {
		for (int i : game_manager.get_available()) {
			cnt += move_probs[i];
			if (cnt > rnd) {
				mcts.update_with_move(-10);
				move = i;
				break;
			}
		}
	}

	r = move;
	return ret;
}

//void MCTSPlayer::get_action(const GameManager& game_manager, array<float, totSize + 1>& r, bool shown, float temp){
//	r = mcts.get_move_probs(game_manager, cout, temp, shown);
//	return;
//}

float MCTSPlayer::get_action(const GameManager& game_manager, pair<int, array<float, totSize + 1> >& r, bool shown, float temp)
{
	int move = totSize;
	float rnd = get_random(0.0f, 1.0f);
	float cnt = 0.0f;
	r.second = mcts.get_move_probs(game_manager, cout, temp, shown);
	float ret = get_win_prob();

	if (_is_selfplay) {
		// get dirichlet distribution
		int n_legal = game_manager.get_n_available();
		float sum = 0.0f;

		for (int i = 0; i < n_legal; ++i) {
			dirichlet[i] = get_gamma();
			sum += dirichlet[i];
		}
		for (int i = 0; i < n_legal; ++i)
			dirichlet[i] /= sum;

		int dnt = 0;
		for (int i : game_manager.get_available()) {
			cnt += r.second[i] * (1.0f - dir_portion) + dirichlet[dnt++] * dir_portion;
			if (cnt > rnd) {
				mcts.update_with_move(i);
				move = i;
				break;
			}
		}
	}
	
	else {
		for (int i : game_manager.get_available()) {
			cnt += r.second[i];
			if (cnt > rnd) {
				mcts.update_with_move(-10);
				move = i;
				break;
			}
		}
	}

	r.first = move;
	return ret;
}

void MCTSPlayer::get_random_action(const GameManager& game_manager, int& r, bool shown, float temp)
{
	//mcts.get_move_probs(game_manager, cout, temp, shown);
	//int idx = static_cast<int>(get_random(0.0f, 1.0f) * game_manager.get_n_available());
	//r = game_manager.get_available()[idx];
	//mcts.update_with_move(r);
	//return;
	array<float, totSize + 1> move_probs = mcts.get_move_probs(game_manager, cout, temp, shown);
	int move = totSize;
	float rnd = get_random(0.0f, 1.0f);
	float cnt = 0.0f;

	if (_is_selfplay) {
		// get dirichlet distribution
		int n_legal = game_manager.get_n_available();
		float sum = 0.0f;

		for (int i = 0; i < n_legal; ++i) {
			dirichlet[i] = get_gamma();
			sum += dirichlet[i];
		}
		for (int i = 0; i < n_legal; ++i)
			dirichlet[i] /= sum;

		int dnt = 0;
		for (int i : game_manager.get_available()) {
			cnt += move_probs[i] * 0.5f + dirichlet[dnt++] * 0.5f;
			if (cnt > rnd) {
				mcts.update_with_move(i);
				move = i;
				break;
			}
		}
	}

	else {
		for (int i : game_manager.get_available()) {
			cnt += move_probs[i];
			if (cnt > rnd) {
				mcts.update_with_move(-10);
				move = i;
				break;
			}
		}
	}

	r = move;
	return;
}

float MCTSPlayer::get_win_prob() {
	if (!mcts.root->_Q)
		cout << "warning! playout hasn't been performed. return 0.0f";
	return mcts.root->_Q;
}







