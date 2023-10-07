// CMakeProject1.cpp : 애플리케이션의 진입점을 정의합니다.

#include "CMakeProject1.h"
#include "PolicyValue.h"
#include "train.h"
using namespace std;

int main() {
	TrainPipeline* training_pipeline = new TrainPipeline("model3bv56050.pt", "model3bv56000.pt", true, 6050);
	/*training_pipeline->policy_evaluate("", false, 50);*/
	training_pipeline->run();
	delete training_pipeline;

	/*TrainPipeline::play("model3b2700.pt", false, 5000, 0.1f, true, true);*/
	/*TrainPipeline::play("model3bv24000.pt", false, 5000, 0.1f, true, true);*/
	/*TrainPipeline::play("model3bv34500.pt", false, 5000, 0.1f, true, true);*/
	/*TrainPipeline::play("model3bv412000.pt", false, 5000, 0.1f, true, true);*/
	/*TrainPipeline::play("model3bv56000.pt", false, 5000, 0.1f, true, true);*/
		
	/*GameManager g = GameManager();
	int x, y, v;
	while (true) {
		cin >> x >> y;
		v = g.make_move(x, y, true);
		cout << x << " " << y << " " << v << endl;
		g.switch_turn();
	}*/

	return 0;
}


