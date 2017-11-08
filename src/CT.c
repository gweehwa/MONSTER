#ifndef FBC_NN_LOGISTICREGRESSION_HPP_
#define FBC_NN_LOGISTICREGRESSION_HPP_

#include <string>
#include <memory>
#include <vector>

namespace ANN {

template<typename T>
class LogisticRegression { // two categories
public:
	LogisticRegression() = default;
	int init(const T* data, const T* labels, int train_num, int feature_length,
		int reg_kinds = -1, T learning_rate = 0.00001, int iterations = 10000, int train_method = 0, int mini_batch_size = 1);
	int train(const std::string& model);
	int load_model(const std::string& model);
	T predict(const T* data, int feature_length) const; // y = 1/(1+exp(-(wx+b)))

	// Regularization kinds
	enum RegKinds {
		REG_DISABLE = -1, // Regularization disabled
		REG_L1 = 0 // L1 norm
	};

	// Training methods
	enum Methods {
		BATCH = 0,
		MINI_BATCH = 1
	};

private:
	int store_model(const std::string& model) const;
	T calc_sigmoid(T x) const; // y = 1/(1+exp(-x))
	T norm(const std::vector<T>& v1, const std::vector<T>& v2) const;
	void batch_gradient_descent();
	void mini_batch_gradient_descent();
	void gradient_descent(const std::vector<std::vector<T>>& data_batch, const std::vector<T>& labels_batch, int length_batch);

	std::vector<std::vector<T>> data;
	std::vector<T> labels;
	int iterations = 1000;
	int train_num = 0; // train samples num
	int feature_length = 0;
	T learning_rate = 0.00001;
	std::vector<T> thetas; // coefficient
	//T epsilon = 0.000001; // termination condition
	T lambda = (T)0.; // regularization method
	int train_method = 0;
	int mini_batch_size = 1;
};

} // namespace ANN

#endif // FBC_NN_LOGISTICREGRESSION_HPP_

#include "logistic_regression.hpp"
#include <fstream>
#include <algorithm>
#include <functional>
#include <numeric>
#include "common.hpp"

namespace ANN {

template<typename T>
int LogisticRegression<T>::init(const T* data, const T* labels, int train_num, int feature_length,
	int reg_kinds, T learning_rate, int iterations, int train_method, int mini_batch_size)
{
	if (train_num < 2) {
		fprintf(stderr, "logistic regression train samples num is too little: %d\n", train_num);
		return -1;
	}
	if (learning_rate <= 0) {
		fprintf(stderr, "learning rate must be greater 0: %f\n", learning_rate);
		return -1;
	}
	if (iterations <= 0) {
		fprintf(stderr, "number of iterations cannot be zero or a negative number: %d\n", iterations);
		return -1;
	}

	CHECK(reg_kinds == -1 || reg_kinds == 0);
	CHECK(train_method == 0 || train_method == 1);
	CHECK(mini_batch_size >= 1 && mini_batch_size < train_num);

	if (reg_kinds == REG_L1) this->lambda = (T)1.;
	if (train_method == MINI_BATCH) this->train_method = 1;
	this->mini_batch_size = mini_batch_size;

	this->learning_rate = learning_rate;
	this->iterations = iterations;

	this->train_num = train_num;
	this->feature_length = feature_length;

	this->data.resize(train_num);
	this->labels.resize(train_num);

	for (int i = 0; i < train_num; ++i) {
		const T* p = data + i * feature_length;
		this->data[i].resize(feature_length+1);
		this->data[i][0] = (T)1; // bias

		for (int j = 0; j < feature_length; ++j) {
			this->data[i][j+1] = p[j];
		}

		this->labels[i] = labels[i];
	}

	this->thetas.resize(feature_length + 1, (T)0.); // bias + feature_length

	return 0;
}

template<typename T>
int LogisticRegression<T>::train(const std::string& model)
{
	CHECK(data.size() == labels.size());

	if (train_method == BATCH) batch_gradient_descent();
	else mini_batch_gradient_descent();

	CHECK(store_model(model) == 0);

	return 0;
}

template<typename T>
void LogisticRegression<T>::batch_gradient_descent()
{
	for (int i = 0; i < iterations; ++i) {
		gradient_descent(data, labels, train_num);

		/*std::unique_ptr<T[]> z(new T[train_num]), gradient(new T[thetas.size()]);
		for (int j = 0; j < train_num; ++j) {
			z.get()[j] = (T)0.;
			for (int t = 0; t < feature_length + 1; ++t) {
				z.get()[j] += data[j][t] * thetas[t];
			}
		}

		std::unique_ptr<T[]> pcal_a(new T[train_num]), pcal_b(new T[train_num]), pcal_ab(new T[train_num]);
		for (int j = 0; j < train_num; ++j) {
			pcal_a.get()[j] = calc_sigmoid(z.get()[j]) - labels[j];
			pcal_b.get()[j] = data[j][0]; // bias
			pcal_ab.get()[j] = pcal_a.get()[j] * pcal_b.get()[j];
		}

		gradient.get()[0] = ((T)1. / train_num) * std::accumulate(pcal_ab.get(), pcal_ab.get() + train_num, (T)0.); // bias

		for (int j = 1; j < thetas.size(); ++j) {
			for (int t = 0; t < train_num; ++t) {
				pcal_b.get()[t] = data[t][j];
				pcal_ab.get()[t] = pcal_a.get()[t] * pcal_b.get()[t];
			}

			gradient.get()[j] = ((T)1. / train_num) * std::accumulate(pcal_ab.get(), pcal_ab.get() + train_num, (T)0.) +
				(lambda / train_num) * thetas[j];
		}

		for (int i = 0; i < thetas.size(); ++i) {
			thetas[i] = thetas[i] - learning_rate / train_num * gradient.get()[i];
		}*/
	}
}

template<typename T>
void LogisticRegression<T>::mini_batch_gradient_descent()
{
	const int step = mini_batch_size;
	const int iter_batch = (train_num + step - 1) / step;

	for (int i = 0; i < iterations; ++i) {
		int pos{ 0 };

		for (int j = 0; j < iter_batch; ++j) {
			std::vector<std::vector<T>> data_batch;
			std::vector<T> labels_batch;
			int remainder{ 0 };

			if (pos + step < train_num) remainder = step;
			else remainder = train_num - pos;

			data_batch.resize(remainder);
			labels_batch.resize(remainder, (T)0.);

			for (int t = 0; t < remainder; ++t) {
				data_batch[t].resize(thetas.size(), (T)0.);
				for (int m = 0; m < thetas.size(); ++m) {
					data_batch[t][m] = data[pos + t][m];
				}

				labels_batch[t] = labels[pos + t];
			}

			gradient_descent(data_batch, labels_batch, remainder);

			pos += step;
		}
	}
}

template<typename T>
void LogisticRegression<T>::gradient_descent(const std::vector<std::vector<T>>& data_batch, const std::vector<T>& labels_batch, int length_batch)
{
	std::unique_ptr<T[]> z(new T[length_batch]), gradient(new T[this->thetas.size()]);
	for (int j = 0; j < length_batch; ++j) {
		z.get()[j] = (T)0.;
		for (int t = 0; t < this->thetas.size(); ++t) {
			z.get()[j] += data_batch[j][t] * this->thetas[t];
		}
	}

	std::unique_ptr<T[]> pcal_a(new T[length_batch]), pcal_b(new T[length_batch]), pcal_ab(new T[length_batch]);
	for (int j = 0; j < length_batch; ++j) {
		pcal_a.get()[j] = calc_sigmoid(z.get()[j]) - labels_batch[j];
		pcal_b.get()[j] = data_batch[j][0]; // bias
		pcal_ab.get()[j] = pcal_a.get()[j] * pcal_b.get()[j];
	}

	gradient.get()[0] = ((T)1. / length_batch) * std::accumulate(pcal_ab.get(), pcal_ab.get() + length_batch, (T)0.); // bias

	for (int j = 1; j < this->thetas.size(); ++j) {
		for (int t = 0; t < length_batch; ++t) {
			pcal_b.get()[t] = data_batch[t][j];
			pcal_ab.get()[t] = pcal_a.get()[t] * pcal_b.get()[t];
		}

		gradient.get()[j] = ((T)1. / length_batch) * std::accumulate(pcal_ab.get(), pcal_ab.get() + length_batch, (T)0.) +
			(this->lambda / length_batch) * this->thetas[j];
	}

	for (int i = 0; i < thetas.size(); ++i) {
		this->thetas[i] = this->thetas[i] - this->learning_rate / length_batch * gradient.get()[i];
	}
}

template<typename T>
int LogisticRegression<T>::load_model(const std::string& model)
{
	std::ifstream file;
	file.open(model.c_str(), std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	int length{ 0 };
	file.read((char*)&length, sizeof(length));
	thetas.resize(length);
	file.read((char*)thetas.data(), sizeof(T)*thetas.size());

	file.close();

	return 0;
}

template<typename T>
T LogisticRegression<T>::predict(const T* data, int feature_length) const
{
	CHECK(feature_length + 1 == thetas.size());

	T value{(T)0.};
	for (int t = 1; t < thetas.size(); ++t) {
		value += data[t - 1] * thetas[t];
	}
	return (calc_sigmoid(value + thetas[0]));
}

template<typename T>
int LogisticRegression<T>::store_model(const std::string& model) const
{
	std::ofstream file;
	file.open(model.c_str(), std::ios::binary);
	if (!file.is_open()) {
		fprintf(stderr, "open file fail: %s\n", model.c_str());
		return -1;
	}

	int length = thetas.size();
	file.write((char*)&length, sizeof(length));
	file.write((char*)thetas.data(), sizeof(T) * thetas.size());

	file.close();

	return 0;
}

template<typename T>
T LogisticRegression<T>::calc_sigmoid(T x) const
{
	return ((T)1 / ((T)1 + exp(-x)));
}

template<typename T>
T LogisticRegression<T>::norm(const std::vector<T>& v1, const std::vector<T>& v2) const
{
	CHECK(v1.size() == v2.size());

	T sum{ 0 };

	for (int i = 0; i < v1.size(); ++i) {
		T minus = v1[i] - v2[i];
		sum += (minus * minus);
	}

	return std::sqrt(sum);
}

template class LogisticRegression<float>;
template class LogisticRegression<double>;

} // namespace ANN

#include "funset.hpp"
#include <iostream>
#include "perceptron.hpp"
#include "BP.hpp""
#include "CNN.hpp"
#include "linear_regression.hpp"
#include "naive_bayes_classifier.hpp"
#include "logistic_regression.hpp"
#include "common.hpp"
#include <opencv2/opencv.hpp>

// ================================ logistic regression =====================
int test_logistic_regression_train()
{
	const std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
	cv::Mat data, labels;

	for (int i = 1; i < 11; ++i) {
		const std::vector<std::string> label{ "0_", "1_" };

		for (const auto& value : label) {
			std::string name = std::to_string(i);
			name = image_path + value + name + ".jpg";

			cv::Mat image = cv::imread(name, 0);
			if (image.empty()) {
				fprintf(stderr, "read image fail: %s\n", name.c_str());
				return -1;
			}

			data.push_back(image.reshape(0, 1));
		}
	}
	data.convertTo(data, CV_32F);

	std::unique_ptr<float[]> tmp(new float[20]);
	for (int i = 0; i < 20; ++i) {
		if (i % 2 == 0) tmp[i] = 0.f;
		else tmp[i] = 1.f;
	}
	labels = cv::Mat(20, 1, CV_32FC1, tmp.get());

	ANN::LogisticRegression<float> lr;
	const float learning_rate{ 0.00001f };
	const int iterations{ 250 };
	int reg_kinds = lr.REG_DISABLE; //ANN::LogisticRegression<float>::REG_L1;
	int train_method = lr.MINI_BATCH; //ANN::LogisticRegression<float>::BATCH;
	int mini_batch_size = 5;

	int ret = lr.init((float*)data.data, (float*)labels.data, data.rows, data.cols/*,
		reg_kinds, learning_rate, iterations, train_method, mini_batch_size*/);
	if (ret != 0) {
		fprintf(stderr, "logistic regression init fail: %d\n", ret);
		return -1;
	}

	const std::string model{ "E:/GitCode/NN_Test/data/logistic_regression.model" };

	ret = lr.train(model);
	if (ret != 0) {
		fprintf(stderr, "logistic regression train fail: %d\n", ret);
		return -1;
	}

	return 0;
}

int test_logistic_regression_predict()
{
	const std::string image_path{ "E:/GitCode/NN_Test/data/images/digit/handwriting_0_and_1/" };
	cv::Mat data, labels, result;

	for (int i = 11; i < 21; ++i) {
		const std::vector<std::string> label{ "0_", "1_" };

		for (const auto& value : label) {
			std::string name = std::to_string(i);
			name = image_path + value + name + ".jpg";

			cv::Mat image = cv::imread(name, 0);
			if (image.empty()) {
				fprintf(stderr, "read image fail: %s\n", name.c_str());
				return -1;
			}

			data.push_back(image.reshape(0, 1));
		}
	}
	data.convertTo(data, CV_32F);

	std::unique_ptr<int[]> tmp(new int[20]);
	for (int i = 0; i < 20; ++i) {
		if (i % 2 == 0) tmp[i] = 0;
		else tmp[i] = 1;
	}
	labels = cv::Mat(20, 1, CV_32SC1, tmp.get());

	CHECK(data.rows == labels.rows);

	const std::string model{ "E:/GitCode/NN_Test/data/logistic_regression.model" };

	ANN::LogisticRegression<float> lr;
	int ret = lr.load_model(model);
	if (ret != 0) {
		fprintf(stderr, "load logistic regression model fail: %d\n", ret);
		return -1;
	}

	for (int i = 0; i < data.rows; ++i) {
		float probability = lr.predict((float*)(data.row(i).data), data.cols);

		fprintf(stdout, "probability: %.6f, ", probability);
		if (probability > 0.5) fprintf(stdout, "predict result: 1, ");
		else fprintf(stdout, "predict result: 0, ");
		fprintf(stdout, "actual result: %d\n", ((int*)(labels.row(i).data))[0]);
	}

	return 0;
}

/*
 * split.Rule = CT
 */
#include <math.h>
#include "causalTree.h"
#include "causalTreeproto.h"

static double *sums, *wtsums, *treatment_effect;
static double *wts, *trs, *trsums;
static int *countn;
static int *tsplit;
static double *wtsqrsums, *trsqrsums;

int
CTinit(int n, double *y[], int maxcat, char **error,
        int *size, int who, double *wt, double *treatment, 
        int bucketnum, int bucketMax, double *train_to_est_ratio)
{
    if (who == 1 && maxcat > 0) {
        graycode_init0(maxcat);
        countn = (int *) ALLOC(2 * maxcat, sizeof(int));
        tsplit = countn + maxcat;
        treatment_effect = (double *) ALLOC(8 * maxcat, sizeof(double));
        wts = treatment_effect + maxcat;
        trs = wts + maxcat;
        sums = trs + maxcat;
        wtsums = sums + maxcat;
        trsums = wtsums + maxcat;
        wtsqrsums = trsums + maxcat;
        trsqrsums = wtsqrsums + maxcat;
    }
    *size = 1;
    *train_to_est_ratio = n * 1.0 / ct.NumHonest;
    return 0;
}

void
CTss(int n, double *y[], double *value, double *con_mean, double *tr_mean, 
     double *risk, double *wt, double *treatment, double *IV, double max_y,
     double alpha, double train_to_est_ratio)
{
    int i;
    double temp0 = 0., temp1 = 0., twt = 0.; /* sum of the weights */ 
    double ttreat = 0.;
    double effect;
    double tr_var, con_var;
    double con_sqr_sum = 0., tr_sqr_sum = 0.;
    double xz_sum = 0., xy_sum = 0., x_sum = 0., y_sum = 0., z_sum = 0.;
    double yz_sum = 0., xx_sum = 0., yy_sum = 0., zz_sum = 0.;
    double alpha_1 = 0., alpha_0 = 0., beta_1 = 0., beta_0 = 0.;
    double numerator, denominator;
    for (i = 0; i < n; i++) {
        temp1 += *y[i] * wt[i] * treatment[i];
        temp0 += *y[i] * wt[i] * (1 - treatment[i]);
        twt += wt[i];
        ttreat += wt[i] * treatment[i];
        tr_sqr_sum += (*y[i]) * (*y[i]) * wt[i] * treatment[i];
        con_sqr_sum += (*y[i]) * (*y[i]) * wt[i] * (1- treatment[i]);
        xz_sum += *y[i] * IV[i];
        xy_sum += treatment[i] * IV[i];
        x_sum += IV[i];
        y_sum += treatment[i];
        z_sum += *y[i];
        yz_sum += *y[i] * treatment[i];
        xx_sum += IV[i] * IV[i];
        yy_sum += treatment[i] * treatment[i];
        zz_sum += *y[i] * *y[i];
    }

    alpha_1 = (n * xz_sum - x_sum * z_sum) / (n * xy_sum - x_sum * y_sum);
    effect = alpha_1;
    alpha_0 = (z_sum - alpha_1 * y_sum) / n;
    beta_1 = (n * xy_sum - x_sum * y_sum) / (n * xx_sum - x_sum * x_sum);
    beta_0 = (y_sum - beta_1 * x_sum) / n;

    *tr_mean = temp1 / ttreat;
    *con_mean = temp0 / (twt - ttreat);
    *value = effect;
    
    numerator = zz_sum + n * alpha_0 * alpha_0 + alpha_1 * alpha_1 * yy_sum - 2 * alpha_0 * z_sum - 2 * alpha_1 * yz_sum + 2 * alpha_0 * alpha_1 * y_sum;
    denominator = n * beta_0 * beta_0 + beta_1 * beta_1 * xx_sum + y_sum * y_sum / n + 2 * beta_0 * beta_1 * x_sum - 2 * beta_0 * y_sum - 2 * beta_1 * x_sum * y_sum / n;
    *risk = 4 * twt * max_y * max_y - alpha * twt * effect * effect + (1 - alpha) * (1 + train_to_est_ratio) * twt * (numerator / denominator);
// PARAMETER!    
    if(abs(n * xy_sum - x_sum * y_sum) <= 0 * n * n){
        effect = temp1 / ttreat - temp0 / (twt - ttreat);  
        *value = effect;
        tr_var = tr_sqr_sum / ttreat - temp1 * temp1 / (ttreat * ttreat);
        con_var = con_sqr_sum / (twt - ttreat) - temp0 * temp0 / ((twt - ttreat) * (twt - ttreat));
        *risk = 4 * twt * max_y * max_y - alpha * twt * effect * effect + 
        (1 - alpha) * (1 + train_to_est_ratio) * twt * (tr_var /ttreat  + con_var / (twt - ttreat));
    }
            
}

void CT(int n, double *y[], double *x, int nclass, int edge, double *improve, double *split, 
        int *csplit, double myrisk, double *wt, double *treatment, double *IV, int minsize, double alpha,
        double train_to_est_ratio)
{
    int i, j;
    double temp;
    double left_sum, right_sum;
    double left_tr_sum, right_tr_sum;
    double left_tr, right_tr;
    double left_wt, right_wt;
    int left_n, right_n;
    double best;
    int direction = LEFT;
    int where = 0;
    double node_effect, left_effect, right_effect;
    double left_temp, right_temp;
    int min_node_size = minsize;
    
    double tr_var, con_var;
    double right_sqr_sum, right_tr_sqr_sum, left_sqr_sum, left_tr_sqr_sum;
    double left_tr_var, left_con_var, right_tr_var, right_con_var;

    right_wt = 0.;
    right_tr = 0.;
    right_sum = 0.;
    right_tr_sum = 0.;
    right_sqr_sum = 0.;
    right_tr_sqr_sum = 0.;
    right_n = n;
    double right_xz_sum = 0., right_xy_sum = 0., right_x_sum = 0., right_y_sum = 0., right_z_sum = 0.;
    double left_xz_sum = 0., left_xy_sum = 0., left_x_sum = 0., left_y_sum = 0., left_z_sum = 0.;
    double right_yz_sum = 0., right_xx_sum = 0., right_yy_sum = 0., right_zz_sum = 0.;
    double left_yz_sum = 0., left_xx_sum = 0., left_yy_sum = 0., left_zz_sum = 0.;
    double alpha_1 = 0., alpha_0 = 0., beta_1 = 0., beta_0 = 0.;
    double numerator, denominator;
    for (i = 0; i < n; i++) {
        right_wt += wt[i];
        right_tr += wt[i] * treatment[i];
        right_sum += *y[i] * wt[i];
        right_tr_sum += *y[i] * wt[i] * treatment[i];
        right_sqr_sum += (*y[i]) * (*y[i]) * wt[i];
        right_tr_sqr_sum += (*y[i]) * (*y[i]) * wt[i] * treatment[i];
        right_xz_sum += *y[i] * IV[i];
        right_xy_sum += treatment[i] * IV[i];
        right_x_sum += IV[i];
        right_y_sum += treatment[i];
        right_z_sum += *y[i];
        right_yz_sum += *y[i] * treatment[i];
        right_xx_sum += IV[i] * IV[i];
        right_yy_sum += treatment[i] * treatment[i];
        right_zz_sum += *y[i] * *y[i];
    }

    alpha_1 = (right_n * right_xz_sum - right_x_sum * right_z_sum) / (right_n * right_xy_sum - right_x_sum * right_y_sum);
    alpha_0 = (right_z_sum - alpha_1 * right_y_sum) / right_n;
    beta_1 = (right_n * right_xy_sum - right_x_sum * right_y_sum) / (right_n * right_xx_sum - right_x_sum * right_x_sum);
    beta_0 = (right_y_sum - beta_1 * right_x_sum) / right_n;
    temp = alpha_1;
    numerator = right_zz_sum + right_n * alpha_0 * alpha_0 + alpha_1 * alpha_1 * right_yy_sum - 2 * alpha_0 * right_z_sum - 2 * alpha_1 * right_yz_sum + 2 * alpha_0 * alpha_1 * right_y_sum;
    denominator = right_n * beta_0 * beta_0 + beta_1 * beta_1 * right_xx_sum + right_y_sum * right_y_sum / right_n + 2 * beta_0 * beta_1 * right_x_sum - 2 * beta_0 * right_y_sum - 2 * beta_1 * right_x_sum * right_y_sum / right_n;
    node_effect = alpha * temp * temp * right_wt - (1 - alpha) * (1 + train_to_est_ratio) 
        * right_wt * (numerator / denominator);
// PARAMETER!        
    if(abs(right_n * right_xy_sum - right_x_sum * right_y_sum) <= 0 * right_n * right_n){
            temp = right_tr_sum / right_tr - (right_sum - right_tr_sum) / (right_wt - right_tr);
            tr_var = right_tr_sqr_sum / right_tr - right_tr_sum * right_tr_sum / (right_tr * right_tr);
            con_var = (right_sqr_sum - right_tr_sqr_sum) / (right_wt - right_tr)
                - (right_sum - right_tr_sum) * (right_sum - right_tr_sum) 
                / ((right_wt - right_tr) * (right_wt - right_tr));
            node_effect = alpha * temp * temp * right_wt - (1 - alpha) * (1 + train_to_est_ratio) 
                * right_wt * (tr_var / right_tr  + con_var / (right_wt - right_tr));
    }
    
    if (nclass == 0) {
        /* continuous predictor */
        left_wt = 0;
        left_tr = 0;
        left_n = 0;
        left_sum = 0;
        left_tr_sum = 0;
        left_sqr_sum = 0;
        left_tr_sqr_sum = 0;
        best = 0;
        
        for (i = 0; right_n > edge; i++) {
            left_wt += wt[i];
            right_wt -= wt[i];
            left_tr += wt[i] * treatment[i];
            right_tr -= wt[i] * treatment[i];
            left_n++;
            right_n--;
            temp = *y[i] * wt[i] * treatment[i];
            left_tr_sum += temp;
            right_tr_sum -= temp;
            left_sum += *y[i] * wt[i];
            right_sum -= *y[i] * wt[i];
            temp = (*y[i]) *  (*y[i]) * wt[i];
            left_sqr_sum += temp;
            right_sqr_sum -= temp;
            temp = (*y[i]) * (*y[i]) * wt[i] * treatment[i];
            left_tr_sqr_sum += temp;
            right_tr_sqr_sum -= temp;
                
            left_xz_sum += *y[i] * IV[i];
            right_xz_sum -= *y[i] * IV[i];
            left_xy_sum += treatment[i] * IV[i];
            right_xy_sum -= treatment[i] * IV[i];
            left_x_sum += IV[i];
            right_x_sum -= IV[i];
            left_y_sum += treatment[i];
            right_y_sum -= treatment[i];
            left_z_sum += *y[i];
            right_z_sum -= *y[i];
            left_yz_sum += *y[i] * treatment[i];
            right_yz_sum -= *y[i] * treatment[i];
            left_xx_sum += IV[i] * IV[i];
            right_xx_sum -= IV[i] * IV[i];
            left_yy_sum += treatment[i] * treatment[i];
            right_yy_sum -= treatment[i] * treatment[i];
            left_zz_sum += *y[i] * *y[i];
            right_zz_sum -= *y[i] * *y[i];
            
            if (x[i + 1] != x[i] && left_n >= edge &&
                (int) left_tr >= min_node_size &&
                (int) left_wt - (int) left_tr >= min_node_size &&
                (int) right_tr >= min_node_size &&
                (int) right_wt - (int) right_tr >= min_node_size) {                             
                                            
                alpha_1 = (left_n * left_xz_sum - left_x_sum * left_z_sum) / (left_n * left_xy_sum - left_x_sum * left_y_sum);
                alpha_0 = (left_z_sum - alpha_1 * left_y_sum) / left_n;
                beta_1 = (left_n * left_xy_sum - left_x_sum * left_y_sum) / (left_n * left_xx_sum - left_x_sum * left_x_sum);
                beta_0 = (left_y_sum - beta_1 * left_x_sum) / left_n;
                left_temp = alpha_1;
                numerator = left_zz_sum + left_n * alpha_0 * alpha_0 + alpha_1 * alpha_1 * left_yy_sum - 2 * alpha_0 * left_z_sum - 2 * alpha_1 * left_yz_sum + 2 * alpha_0 * alpha_1 * left_y_sum;
                denominator = left_n * beta_0 * beta_0 + beta_1 * beta_1 * left_xx_sum + left_y_sum * left_y_sum / left_n + 2 * beta_0 * beta_1 * left_x_sum - 2 * beta_0 * left_y_sum - 2 * beta_1 * left_x_sum * left_y_sum / left_n;
                left_effect = alpha * left_temp * left_temp * left_wt - (1 - alpha) * (1 + train_to_est_ratio) 
                    * left_wt * (numerator / denominator);
// PARAMETER!                    
                if(abs(left_n * left_xy_sum - left_x_sum * left_y_sum) <= 0 * left_n * left_n){
                left_temp = left_tr_sum / left_tr - (left_sum - left_tr_sum) / (left_wt - left_tr);
                left_tr_var = left_tr_sqr_sum / left_tr - 
                    left_tr_sum  * left_tr_sum / (left_tr * left_tr);
                left_con_var = (left_sqr_sum - left_tr_sqr_sum) / (left_wt - left_tr)  
                    - (left_sum - left_tr_sum) * (left_sum - left_tr_sum)
                    / ((left_wt - left_tr) * (left_wt - left_tr));        
                left_effect = alpha * left_temp * left_temp * left_wt
                        - (1 - alpha) * (1 + train_to_est_ratio) * left_wt 
                    * (left_tr_var / left_tr + left_con_var / (left_wt - left_tr));}
                

                alpha_1 = (right_n * right_xz_sum - right_x_sum * right_z_sum) / (right_n * right_xy_sum - right_x_sum * right_y_sum);
                alpha_0 = (right_z_sum - alpha_1 * right_y_sum) / right_n;
                beta_1 = (right_n * right_xy_sum - right_x_sum * right_y_sum) / (right_n * right_xx_sum - right_x_sum * right_x_sum);
                beta_0 = (right_y_sum - beta_1 * right_x_sum) / right_n;
                right_temp = alpha_1;
                numerator = right_zz_sum + right_n * alpha_0 * alpha_0 + alpha_1 * alpha_1 * right_yy_sum - 2 * alpha_0 * right_z_sum - 2 * alpha_1 * right_yz_sum + 2 * alpha_0 * alpha_1 * right_y_sum;
                denominator = right_n * beta_0 * beta_0 + beta_1 * beta_1 * right_xx_sum + right_y_sum * right_y_sum / right_n + 2 * beta_0 * beta_1 * right_x_sum - 2 * beta_0 * right_y_sum - 2 * beta_1 * right_x_sum * right_y_sum / right_n;
                right_effect = alpha * right_temp * right_temp * right_wt - (1 - alpha) * (1 + train_to_est_ratio) 
                    * right_wt * (numerator / denominator);
// PARAMETER!                    
                if(abs(right_n * right_xy_sum - right_x_sum * right_y_sum) <= 0 * right_n * right_n){
                right_temp = right_tr_sum / right_tr - (right_sum - right_tr_sum) / (right_wt - right_tr);
                right_tr_var = right_tr_sqr_sum / right_tr -
                    right_tr_sum * right_tr_sum / (right_tr * right_tr);
                right_con_var = (right_sqr_sum - right_tr_sqr_sum) / (right_wt - right_tr)
                    - (right_sum - right_tr_sum) * (right_sum - right_tr_sum) 
                    / ((right_wt - right_tr) * (right_wt - right_tr));
                right_effect = alpha * right_temp * right_temp * right_wt
                        - (1 - alpha) * (1 + train_to_est_ratio) * right_wt * 
                            (right_tr_var / right_tr + right_con_var / (right_wt - right_tr));}
                

                
                temp = left_effect + right_effect - node_effect;
                if (temp > best) {
                    best = temp;
                    where = i;               
                    if (left_temp < right_temp){
                        direction = LEFT;
                    }
                    else{
                        direction = RIGHT;
                    }
                }             
            }
        }
        
        *improve = best;
        if (best > 0) {         /* found something */
        csplit[0] = direction;
            *split = (x[where] + x[where + 1]) / 2; 
        }
    }
    
    /*
    * Categorical predictor
    */
    else {
        for (i = 0; i < nclass; i++) {
            countn[i] = 0;
            wts[i] = 0;
            trs[i] = 0;
            sums[i] = 0;
            wtsums[i] = 0;
            trsums[i] = 0;
            wtsqrsums[i] = 0;
            trsqrsums[i] = 0;
        }
        
        /* rank the classes by treatment effect */
        for (i = 0; i < n; i++) {
            j = (int) x[i] - 1;
            countn[j]++;
            wts[j] += wt[i];
            trs[j] += wt[i] * treatment[i];
            sums[j] += *y[i];
            wtsums[j] += *y[i] * wt[i];
            trsums[j] += *y[i] * wt[i] * treatment[i];
            wtsqrsums[j] += (*y[i]) * (*y[i]) * wt[i];
            trsqrsums[j] +=  (*y[i]) * (*y[i]) * wt[i] * treatment[i];
        }
        
        for (i = 0; i < nclass; i++) {
            if (countn[i] > 0) {
                tsplit[i] = RIGHT;
                treatment_effect[i] = trsums[j] / trs[j] - (wtsums[j] - trsums[j]) / (wts[j] - trs[j]);
            } else
                tsplit[i] = 0;
        }
        graycode_init2(nclass, countn, treatment_effect);
        
        /*
         * Now find the split that we want
         */
        
        left_wt = 0;
        left_tr = 0;
        left_n = 0;
        left_sum = 0;
        left_tr_sum = 0;
        left_sqr_sum = 0.;
        left_tr_sqr_sum = 0.;
        
        best = 0;
        where = 0;
        while ((j = graycode()) < nclass) {
            tsplit[j] = LEFT;
            left_n += countn[j];
            right_n -= countn[j];
            
            left_wt += wts[j];
            right_wt -= wts[j];
            
            left_tr += trs[j];
            right_tr -= trs[j];
            
            left_sum += wtsums[j];
            right_sum -= wtsums[j];
            
            left_tr_sum += trsums[j];
            right_tr_sum -= trsums[j];
            
            left_sqr_sum += wtsqrsums[j];
            right_sqr_sum -= wtsqrsums[j];
            
            left_tr_sqr_sum += trsqrsums[j];
            right_tr_sqr_sum -= trsqrsums[j];
            
            if (left_n >= edge && right_n >= edge &&
                (int) left_tr >= min_node_size &&
                (int) left_wt - (int) left_tr >= min_node_size &&
                (int) right_tr >= min_node_size &&
                (int) right_wt - (int) right_tr >= min_node_size) {
                
                left_temp = left_tr_sum / left_tr - (left_sum - left_tr_sum) 
                    / (left_wt - left_tr);
                
                left_tr_var = left_tr_sqr_sum / left_tr 
                    - left_tr_sum  * left_tr_sum / (left_tr * left_tr);
                left_con_var = (left_sqr_sum - left_tr_sqr_sum) / (left_wt - left_tr)  
                    - (left_sum - left_tr_sum) * (left_sum - left_tr_sum)
                    / ((left_wt - left_tr) * (left_wt - left_tr));       
                left_effect = alpha * left_temp * left_temp * left_wt
                    - (1 - alpha) * (1 + train_to_est_ratio) * left_wt * 
                        (left_tr_var / left_tr + left_con_var / (left_wt - left_tr));
                
                right_temp = right_tr_sum / right_tr - (right_sum - right_tr_sum) 
                    / (right_wt - right_tr);
                right_tr_var = right_tr_sqr_sum / right_tr 
                    - right_tr_sum * right_tr_sum / (right_tr * right_tr);
                right_con_var = (right_sqr_sum - right_tr_sqr_sum) / (right_wt - right_tr)
                    - (right_sum - right_tr_sum) * (right_sum - right_tr_sum) 
                    / ((right_wt - right_tr) * (right_wt - right_tr));
                right_effect = alpha * right_temp * right_temp * right_wt
                        - (1 - alpha) * (1 + train_to_est_ratio) * right_wt *
                            (right_tr_var / right_tr + right_con_var / (right_wt - right_tr));
                temp = left_effect + right_effect - node_effect;
            
                
                if (temp > best) {
                    best = temp;
                    
                    if (left_temp > right_temp)
                        for (i = 0; i < nclass; i++) csplit[i] = -tsplit[i];
                    else
                        for (i = 0; i < nclass; i++) csplit[i] = tsplit[i];
                }
            }
        }
        *improve = best;
    }
}


double
    CTpred(double *y, double wt, double treatment, double *yhat, double propensity)
    {
        double ystar;
        double temp;
        
        ystar = y[0] * (treatment - propensity) / (propensity * (1 - propensity));
        temp = ystar - *yhat;
        return temp * temp * wt;
    }
