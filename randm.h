#pragma once
#include <random>
using namespace std;

static random_device rnd_device;
static mt19937 mt;
static const float alpha = 0.3f;

static std::uniform_real_distribution<float> unif_dist(0.0f, 1.0f);
static std::gamma_distribution<float> gamma(alpha);

static float get_random(float s, float e) {
	return unif_dist(rnd_device) * (e - s) + s;
}

static float get_gamma() {
	return gamma(rnd_device);
}