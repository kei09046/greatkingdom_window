// CMakeProject1.cpp : 애플리케이션의 진입점을 정의합니다.

#include "CMakeProject1.h"
#include "PolicyValue.h"
#include "train.h"
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main() {
	TrainPipeline* training_pipeline = new TrainPipeline("model3bv768500.pt", "model3bv768500.pt", /*gpu=*/true, 68500);
	training_pipeline->run(/*is_shown=*/false, 1.0f);
	delete training_pipeline;
	 
	/*TrainPipeline::play("model3b2700.pt", false, 5000, 0.1f, true, true);*/
	/*TrainPipeline::play("model3bv24000.pt", false, 5000, 0.1f, true, true);*/
	/*TrainPipeline::play("model3bv34500.pt", false, 5000, 0.1f, true, true);*/
	/*TrainPipeline::play("model3bv412000.pt", false, 5000, 0.1f, true, true);*/
	/*TrainPipeline::play("model3bv56000.pt", false, 5000, 0.1f, true, true);*/
	//TrainPipeline::play("model3bv646000.pt", true, 10000, 0.1f, true, true);
	//TrainPipeline::play("model3bv748400.pt", false, 20000, 0.1f, true, true); // mc method
	//TrainPipeline::play("model3bv768500.pt", false, 20000, 0.1f, true, true); // td method
	// 
	//ofstream totalResult("Result60000.txt");
	//for (int i = 0; i < 4; ++i) {
	//	ofstream eachResult("res" + to_string(60000 + 2000 * (i + 1)) + ".txt");
	//	string file1 = "model3bv7" + to_string(60000 + 2000 * i) + string(".pt");
	//	string file2 = "model3bv7" + to_string(60000 + 2000 * (i + 1)) + string(".pt");

	//	totalResult << "test model : " << file1 << " to " << file2 << endl;
	//	TrainPipeline::policy_evaluate(file2, file1, totalResult, eachResult, true, true, 0.5f, 20);
	//	totalResult << endl << endl;
	//	eachResult.close();
	//}

	//totalResult.close();
	//return 0;
}


