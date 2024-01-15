#include "memory.h"
using namespace std;


GameData::GameData(array<float, 7 * largeSize>& _state,
	array<float, totSize + 1>& _mcts_probs, float _winner, array<float, 7 * largeSize>& _next_state) :
	state(_state), mcts_probs(_mcts_probs), winner(_winner), next_state(_next_state) {}

GameData::GameData(array<float, 7 * largeSize>&& _state,
	array<float, totSize + 1>&& _mcts_probs, float _winner, array<float, 7 * largeSize>&& _next_state) :
	state(_state), mcts_probs(_mcts_probs), winner(_winner), next_state(_next_state) {}

GameData::GameData() {}


PackedData::PackedData(int idx, float diff, GameData* ptr): idx(idx), diff(diff), gd(ptr)
{}

PackedData::PackedData(): idx(0), diff(0.0f), gd(nullptr) {}

SumTree::SumTree()
{
	for (int i = 0; i < capacity * 2 - 1; ++i)
		tree[i] = 0.0f;
	for (int i = 0; i < capacity; ++i)
		datas[i] = nullptr;
}

void SumTree::propagate(int idx, float delta)
{
	int parent = (idx - 1) / 2;
	tree[parent] += delta;
	if (parent)
		propagate(parent, delta);
	return;
}

void SumTree::update(int idx, float prior)
{
	float d = prior - tree[idx];
	tree[idx] = prior;
	propagate(idx, d);
	return;
}

void SumTree::emplace_back(float prior, std::array<float, 7 * largeSize>& _state, 
	std::array<float, totSize + 1>& _mcts_probs, float _winner, std::array<float, 7 * largeSize>& _next_state)
{
	int idx = loc + capacity - 1;
	if (!(datas[loc] == nullptr))
		delete datas[loc];

	datas[loc++] = new GameData(_state, _mcts_probs, _winner, _next_state);
	update(idx, prior);
	loc %= capacity;
	if (n_elements < capacity)
		n_elements++;

	return;
}

void SumTree::emplace_back(float prior, std::array<float, 7 * largeSize>&& _state, 
	std::array<float, totSize + 1>&& _mcts_probs, float _winner, std::array<float, 7 * largeSize>&& _next_state)
{
	int idx = loc + capacity - 1;
	if (!(datas[loc] == nullptr))
		delete datas[loc];

	datas[loc++] = new GameData(_state, _mcts_probs, _winner, _next_state);
	update(idx, prior);
	loc %= capacity;
	if (n_elements < capacity)
		n_elements++;

	return;
}

int SumTree::find(int idx, float s) const
{
	int left = (idx << 1) + 1;

	if (left >= 2 * capacity - 1)
		return idx;
	if (s <= tree[left])
		return find(left, s);
	return find(left + 1, s - tree[left]);
}

float SumTree::total() const
{
	return tree[0];
}

PackedData SumTree::get(float s) const
{
	int idx = find(0, s);
	int dataIdx = idx - capacity + 1;
	return PackedData(idx, tree[idx], datas[dataIdx]);
}

Memory::Memory() {}

float Memory::get_priority(float delta) const{
	return pow(abs(delta) + ep, alpha);
}

void Memory::emplace_back(float delta, std::array<float, 7 * largeSize>& _state, std::array<float, totSize + 1>& _mcts_probs, float _winner, std::array<float, 7 * largeSize>& _next_state)
{
	sum_tree.emplace_back(get_priority(delta), _state, _mcts_probs, _winner, _next_state);
	return;
}

void Memory::emplace_back(float delta, std::array<float, 7 * largeSize>&& _state,
	std::array<float, totSize + 1>&& _mcts_probs, float _winner, std::array<float, 7 * largeSize>&& _next_state) {
	sum_tree.emplace_back(get_priority(delta), _state, _mcts_probs, _winner, _next_state);
	return;
}

std::array<PackedData, batchSize> Memory::sample()
{
	std::array<PackedData, batchSize> ret;
	float segment = static_cast<float>(sum_tree.total()) / batchSize;
	beta = min(1.0f, beta + beta_increment_per_sampling);
	float rd, max = -1.0f;

	for (int i = 0; i < batchSize; ++i) {
		rd = get_random(segment * i, segment * (i + 1));
		ret[i] = sum_tree.get(rd);
		ret[i].diff = pow(sum_tree.n_elements * ret[i].diff / sum_tree.total(), -beta);
		if (ret[i].diff > max)
			max = ret[i].diff;
	}

	for (int i = 0; i < batchSize; ++i)
		ret[i].diff /= max;

	return ret;
}

void Memory::update(int idx, float delta)
{
	sum_tree.update(idx, get_priority(delta));
}

