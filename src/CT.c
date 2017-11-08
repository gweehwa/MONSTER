#pragma once
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <climits>
#include <math.h>
#include <time.h>

# define VERSION       "V0.10"
# define VERSION_DATE  "2012-6-12"

using namespace std;

const long  double min_threshold=1e-300;

struct sparse_feat                       //稀疏特征表示结构
{
    vector<int> id_vec;
    vector<float> value_vec;
};

class LR                                 //logistic regression实现类
{
private:
    vector<sparse_feat> samp_feat_vec;
    vector<int> samp_class_vec;
    int feat_set_size;
    int class_set_size;
    vector< vector<float> > omega; //模型的参数矩阵omega = feat_set_size * class_set_size
     
public:
    LR();
    ~LR();
    void save_model(string model_file);
    void load_model(string model_file);
    void load_training_file(string training_file);
    void init_omega();
    
    int train_online(int max_loop, double loss_thrd, float learn_rate, float lambda, int avg);    //logistic regression随机梯度优化算法
    vector<float> calc_score(sparse_feat &samp_feat);
    vector<float> score_to_prb(vector<float> &score);
    int score_to_class(vector<float> &score);
    
    float classify_testing_file(string testing_file, string output_file, int output_format);      //模型分类预测

private:
    void read_samp_file(string samp_file, vector<sparse_feat> &samp_feat_vec, vector<int> &samp_class_vec);   //更新函数
    void update_online_ce(int samp_class, sparse_feat &samp_feat, float learn_rate, float lambda);
    void calc_loss_ce(double *loss, float *acc);                                                              //计算损失函数
    float calc_acc(vector<int> &test_class_vec, vector<int> &pred_class_vec);    
    float sigmoid(float x);
    vector<string> string_split(string terms_str, string spliting_tag);

};
#include "LR.h"

LR::LR()
{

}

LR::~LR()
{
}

void LR::save_model(string model_file)
{ 
    cout << "Saving model..." << endl;
    ofstream fout(model_file.c_str());
    for (int k = 0; k < feat_set_size; k++) {
        for (int j = 0; j < class_set_size; j++) {
            fout << omega[k][j] << " ";
        }
        fout << endl;
    }
    fout.close();
}


void LR::load_model(string model_file)
{
    cout << "Loading model..." << endl;
    omega.clear();
    ifstream fin(model_file.c_str());
    if(!fin) {
        cerr << "Error opening file: " << model_file << endl;
        exit(0);
    }
    string line_str;
    while (getline(fin, line_str)) {
        vector<string> line_vec = string_split(line_str, " ");
        vector<float>  line_omega;
        for (vector<string>::iterator it = line_vec.begin(); it != line_vec.end(); it++) {
            float weight = (float)atof(it->c_str());
            line_omega.push_back(weight);
        }
        omega.push_back(line_omega);
    }
    fin.close();
    feat_set_size = (int)omega.size();
    class_set_size = (int)omega[0].size();
}


void LR::read_samp_file(string samp_file, vector<sparse_feat> &samp_feat_vec, vector<int> &samp_class_vec) {
    ifstream fin(samp_file.c_str());
    if(!fin) {
        cerr << "Error opening file: " << samp_file << endl;
        exit(0);
    }
    string line_str;
    while (getline(fin, line_str)) 
    {
        size_t class_pos = line_str.find_first_of("\t");
        int class_id = atoi(line_str.substr(0, class_pos).c_str());
        samp_class_vec.push_back(class_id);
        string terms_str = line_str.substr(class_pos+1);
        sparse_feat samp_feat;
        samp_feat.id_vec.push_back(0); // bias
        samp_feat.value_vec.push_back(1); // bias
        if (terms_str != "") 
        {
            vector<string> fv_vec = string_split(terms_str, " ");
            for (vector<string>::iterator it = fv_vec.begin(); it != fv_vec.end(); it++) 
            {
                size_t feat_pos = it->find_first_of(":");
                int feat_id = atoi(it->substr(0, feat_pos).c_str());
                float feat_value = (float)atof(it->substr(feat_pos+1).c_str());
                samp_feat.id_vec.push_back(feat_id);
                samp_feat.value_vec.push_back(feat_value);
            }
        }
        samp_feat_vec.push_back(samp_feat);
    }
    fin.close();
}


void LR::load_training_file(string training_file)
{
    cout << "Loading training data..." << endl;
    read_samp_file(training_file, samp_feat_vec, samp_class_vec);
    feat_set_size = 0;
    class_set_size = 0;
    for (size_t i = 0; i < samp_class_vec.size(); i++) {
        if (samp_class_vec[i] > class_set_size) {
            class_set_size = samp_class_vec[i];
        }
        if (samp_feat_vec[i].id_vec.back() > feat_set_size) {
            feat_set_size = samp_feat_vec[i].id_vec.back();
        }    
    }
    class_set_size += 1;
    feat_set_size += 1;
}

void LR::init_omega()
{
     float init_value = 0.0;
    //float init_value = (float)1/class_set_size;
    for (int i = 0; i < feat_set_size; i++) 
    {
        vector<float> temp_vec(class_set_size, init_value);
        omega.push_back(temp_vec);
    }
}

// Stochastic Gradient Descent (SGD) optimization for the criteria of  Cross Entropy (CE)
int LR::train_online( int max_loop, double loss_thrd, float learn_rate, float lambda, int avg)
{
    int id = 0;
    double loss = 0.0;
    double loss_pre = 0.0;
    vector< vector<float> > omega_pre=omega;
    float acc=0.0;

    vector< vector<float> > omega_sum(omega);
    
    while (id <= max_loop*(int)samp_class_vec.size()) 
    {
    
        
        if (id%samp_class_vec.size() == 0)    // 完成一次迭代，预处理工作。
        {
            int loop = id/(int)samp_class_vec.size();               //check loss
            loss = 0.0;
            acc = 0.0;
            
            calc_loss_ce(&loss, &acc);    
            cout.setf(ios::left);
            cout << "Iter: " << setw(8) << loop << "Loss: " << setw(18) << loss << "Acc: " << setw(8) << acc << endl;
            if ((loss_pre - loss) < loss_thrd && loss_pre >= loss && id != 0)
            {
                cout << "Reaching the minimal loss value decrease!" << endl;
                break;
            }
            loss_pre = loss;
                                
            if (id)            //表示第一次不做正则项计算
            {
                for (int i=0;i!=omega_pre.size();i++)
                    for (int j=0;j!=omega_pre[i].size();j++)
                        omega[i][j]+=omega_pre[i][j]*lambda;
            }

            omega_pre=omega;
        }

        // update omega
        int r = (int)(rand()%samp_class_vec.size());
        sparse_feat samp_feat = samp_feat_vec[r];
        int samp_class = samp_class_vec[r];
    
         
       update_online_ce(samp_class, samp_feat, learn_rate, lambda);
        
        if (avg == 1 && id%samp_class_vec.size() == 0) 
        {
            for (int i = 0; i < feat_set_size; i++) 
            {
                for (int j = 0; j < class_set_size; j++) 
                {
                    omega_sum[i][j] += omega[i][j];
                }
            }            
        }
        id++;
    }

    if (avg == 1) 
    {
        for (int i = 0; i < feat_set_size; i++) 
        {
            for (int j = 0; j < class_set_size; j++)
            {
                omega[i][j] = (float)omega_sum[i][j] / id;
            }
        }        
    }
    return 0;
}

void LR::update_online_ce(int samp_class, sparse_feat &samp_feat, float learn_rate, float lambda)
{
    
    vector<float> score=calc_score(samp_feat);//(W'*X)
    vector<float> softMaxVec(class_set_size);

    float maxScore=*(max_element(score.begin(),score.end()));
    float softMaxSum=0;

    for (int j=0;j<class_set_size;j++)
    {
        softMaxVec[j]=exp(score[j]-maxScore);
        softMaxSum+=softMaxVec[j];                             //同时除最大的score;
    }
    for (int k=0;k<class_set_size;k++)
        softMaxVec[k]/=softMaxSum;


    for (int i=0;i<class_set_size;i++)                          //对于每一个类
    {
        float error_term=((int)(i==samp_class)-softMaxVec[i]);
        for (int j=0;j<samp_feat.id_vec.size();j++)             //对于每个类中的
        {
            int feat_id=samp_feat.id_vec[j];
            float feat_value=samp_feat.value_vec[j];
            float delt=error_term*feat_value;
            //float regul = lambda * omega[feat_id][i];
            omega[feat_id][i]+=learn_rate*delt;

        }

    }
    
     
}

void LR::calc_loss_ce(double *loss, float *acc)
{
    double loss_value = 0.0;
    int err_num = 0;

    for (size_t i = 0; i < samp_class_vec.size(); i++) 
    {
        int samp_class = samp_class_vec[i];
        sparse_feat samp_feat = samp_feat_vec[i];

        vector<float> score = calc_score(samp_feat);
        vector<float> softMaxVec(class_set_size);
        float softMaxSum=0;

        int pred_class = score_to_class(score);
        if (pred_class != samp_class) 
        {
            err_num += 1;
        }

        float maxScore=*(max_element(score.begin(),score.end()));
        for (int k=0;k<class_set_size;k++)
        {
           softMaxVec[k]=exp(score[k]-maxScore);
           softMaxSum+=softMaxVec[k];                     //同时除最大的score;
        }

        for (int j = 0; j < class_set_size; j++)
        {
            if (j == samp_class) 
            { 

                double yi=softMaxVec[j]/softMaxSum;
                long double temp=yi<min_threshold ? min_threshold:yi;
                loss_value -= log(temp);                             
        
            }


        }
    }

    *loss = loss_value ;
    *acc = 1 - (float)err_num / samp_class_vec.size();
}

vector<float> LR::calc_score(sparse_feat &samp_feat)
{
    vector<float> score(class_set_size, 0);
    for (int j = 0; j < class_set_size; j++)
    {
        for (size_t k = 0; k < samp_feat.id_vec.size(); k++) 
        {
            int feat_id = samp_feat.id_vec[k];
            float feat_value = samp_feat.value_vec[k];
            score[j] += omega[feat_id][j] * feat_value;
        }
    }
    return score;
}

vector<float> LR::score_to_prb(vector<float> &score)
{   
    //vector<float> prb(class_set_size, 0);
    //for (int i = 0; i < class_set_size; i++) 
    //{
    //    float delta_prb_sum = 0.0;
    //    for (int j = 0; j < class_set_size; j++) 
    //    {
    //        delta_prb_sum += exp(score[j] - score[i]);
    //    }
    //    prb[i] = 1 / delta_prb_sum;
    //}
    //return prb;

    vector<float> softMaxVec(class_set_size);

    float maxScore=*(max_element(score.begin(),score.end()));
    float softMaxSum=0;

    for (int j=0;j<class_set_size;j++)
    {
        softMaxVec[j]=exp(score[j]-maxScore);
        softMaxSum+=softMaxVec[j];                             //同时除最大的score;
    }
    for (int k=0;k<class_set_size;k++)
        softMaxVec[k]/=softMaxSum;

    return softMaxVec;

}


int LR::score_to_class(vector<float> &score)
{
    int pred_class = 0;    
    float max_score = score[0];
    for (int j = 1; j < class_set_size; j++) {
        if (score[j] > max_score) {
            max_score = score[j];
            pred_class = j;
        }
    }
    return pred_class;
}

float LR::classify_testing_file(string testing_file, string output_file, int output_format)
{
    cout << "Classifying testing file..." << endl;
    vector<sparse_feat> test_feat_vec;
    vector<int> test_class_vec;
    vector<int> pred_class_vec;
    read_samp_file(testing_file, test_feat_vec, test_class_vec);
    ofstream fout(output_file.c_str());
    for (size_t i = 0; i < test_class_vec.size(); i++) 
    {
        int samp_class = test_class_vec[i];
        sparse_feat samp_feat = test_feat_vec[i];
        vector<float> pred_score = calc_score(samp_feat);            
        int pred_class = score_to_class(pred_score);
        pred_class_vec.push_back(pred_class);
        fout << pred_class << "\t"<<samp_class<<"\t";
        if (output_format == 1) 
        {
            for (int j = 0; j < class_set_size; j++) 
            {
                fout << j << ":" << pred_score[j] << ' '; 
            }        
        }
        else if (output_format == 2) 
        {
            vector<float> pred_prb = score_to_prb(pred_score);
            for (int j = 0; j < class_set_size; j++)
            {
                fout << j << ":" << pred_prb[j] << ' '; 
            }
        }

        fout << endl;        
    }
    fout.close();
    float acc = calc_acc(test_class_vec, pred_class_vec);
    return acc;
}

float LR::calc_acc(vector<int> &test_class_vec, vector<int> &pred_class_vec)
{
    size_t len = test_class_vec.size();
    if (len != pred_class_vec.size()) {
        cerr << "Error: two vectors should have the same lenght." << endl;
        exit(0);
    }
    int err_num = 0;
    for (size_t id = 0; id != len; id++) {
        if (test_class_vec[id] != pred_class_vec[id]) {
            err_num++;
        }
    }
    return 1 - ((float)err_num) / len;
}

float LR::sigmoid(float x) 
{
    double sgm = 1 / (1+exp(-(double)x));
    return (float)sgm;
}

vector<string> LR::string_split(string terms_str, string spliting_tag)
{
    vector<string> feat_vec;
    size_t term_beg_pos = 0;
    size_t term_end_pos = 0;
    while ((term_end_pos = terms_str.find_first_of(spliting_tag, term_beg_pos)) != string::npos) 
    {
        if (term_end_pos > term_beg_pos)
        {
            string term_str = terms_str.substr(term_beg_pos, term_end_pos - term_beg_pos);
            feat_vec.push_back(term_str);
        }
        term_beg_pos = term_end_pos + 1;
    }
    if (term_beg_pos < terms_str.size())
    {
        string end_str = terms_str.substr(term_beg_pos);
        feat_vec.push_back(end_str);
    }
    return feat_vec;
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
