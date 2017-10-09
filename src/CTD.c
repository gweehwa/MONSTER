/*
 * The discrte version for CT
 */
#include "causalTree.h"
#include "causalTreeproto.h"

#ifndef min
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef max
#define max(a,b)  (((a) > (b)) ? (a) : (b))
#endif

static double *sums, *wtsums, *treatment_effect;
static double *wts, *trs, *trsums;
static int *countn;
static int *tsplit;
static double *wtsqrsums, *wttrsqrsums;

// for discrete version:
static int *n_bucket, *n_tr_bucket, *n_con_bucket;
static double *wts_bucket, *trs_bucket;
static double *wtsums_bucket, *trsums_bucket;
static double *wtsqrsums_bucket, *trsqrsums_bucket; 
static double *tr_end_bucket, *con_end_bucket;

static double *xz_sum_bucket;
static double *xy_sum_bucket;
static double *x_sum_bucket;
static double *y_sum_bucket;
static double *z_sum_bucket;
static double *yz_sum_bucket;
static double *xx_sum_bucket;
static double *yy_sum_bucket;
static double *zz_sum_bucket;
static double *x1x1_sum_bucket;
static double *x1x2_sum_bucket;
static double *x1x3_sum_bucket;
static double *x1x4_sum_bucket;
static double *x2x1_sum_bucket;
static double *x2x2_sum_bucket;
static double *x2x3_sum_bucket;
static double *x2x4_sum_bucket;
static double *x3x1_sum_bucket;
static double *x3x2_sum_bucket;
static double *x3x3_sum_bucket;
static double *x3x4_sum_bucket;
static double *x4x1_sum_bucket;
static double *x4x2_sum_bucket;
static double *x4x3_sum_bucket;
static double *x4x4_sum_bucket;
static double *x1y_sum_bucket;
static double *x2y_sum_bucket; 
static double *x3y_sum_bucket;
static double *x4y_sum_bucket;

int
CTDinit(int n, double *y[], int maxcat, char **error,
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
		wttrsqrsums = wtsqrsums + maxcat;
	}
	*size = 1;
	*train_to_est_ratio = n * 1.0 / ct.NumHonest;
	if (bucketnum == 0) 
		Rprintf("ERROR for buket!\n");
	return 0;
}


void
CTDss(int n, double *y[], double *value, double *con_mean, double *tr_mean, 
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
    double x1x1_sum = 0., x1x2_sum = 0., x1x3_sum = 0., x1x4_sum = 0., x2x1_sum = 0., x2x2_sum = 0., x2x3_sum = 0., x2x4_sum = 0., x3x1_sum = 0., x3x2_sum = 0., x3x3_sum = 0., x3x4_sum = 0., x4x1_sum = 0., x4x2_sum = 0., x4x3_sum = 0., x4x4_sum = 0.;
    double x1y_sum = 0., x2y_sum = 0., x3y_sum = 0., x4y_sum = 0.;
    float m[16], inv[16], invOut[16];
    double det;
    double bhat_0 = 0., bhat_1 = 0., bhat_2 = 0., bhat_3 = 0.;
    double error2 = 0., var3 = 0.;	
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
	x1x1_sum += 1 * 1;
        x1x2_sum += 1 * treatment[i];
        x1x3_sum += 1 * IV[i];   
        x1x4_sum += 1 * IV[i] * treatment[i]; 
        x2x1_sum += treatment[i] * 1;
        x2x2_sum += treatment[i] * treatment[i];
        x2x3_sum += treatment[i] * IV[i];   
        x2x4_sum += treatment[i] * IV[i] * treatment[i]; 
        x3x1_sum += IV[i] * 1;
        x3x2_sum += IV[i] * treatment[i];
        x3x3_sum += IV[i] * IV[i];   
        x3x4_sum += IV[i] * IV[i] * treatment[i];  
        x4x1_sum += IV[i] * treatment[i] * 1; 
        x4x2_sum += IV[i] * treatment[i] * treatment[i];
        x4x3_sum += IV[i] * treatment[i] * IV[i];   
        x4x4_sum += IV[i] * treatment[i] * IV[i] * treatment[i];  
        x1y_sum += *y[i];
        x2y_sum += *y[i] * treatment[i];
        x3y_sum += *y[i] * IV[i];  
        x4y_sum += *y[i] * IV[i] * treatment[i];  
    }

    m[0] = x1x1_sum;
    m[1] = x1x2_sum;
    m[2] = x1x3_sum;
    m[3] = x1x4_sum;
    m[4] = x2x1_sum;
    m[5] = x2x2_sum;
    m[6] = x2x3_sum;
    m[7] = x2x4_sum;
    m[8] = x3x1_sum;
    m[9] = x3x2_sum;
    m[10] = x3x3_sum;
    m[11] = x3x4_sum;
    m[12] = x4x1_sum;
    m[13] = x4x2_sum;
    m[14] = x4x3_sum;     
    m[15] = x4x4_sum;   
    inv[0] = m[5]  * m[10] * m[15] - 
             m[5]  * m[11] * m[14] - 
             m[9]  * m[6]  * m[15] + 
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] - 
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] + 
              m[4]  * m[11] * m[14] + 
              m[8]  * m[6]  * m[15] - 
              m[8]  * m[7]  * m[14] - 
              m[12] * m[6]  * m[11] + 
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] - 
             m[4]  * m[11] * m[13] - 
             m[8]  * m[5] * m[15] + 
             m[8]  * m[7] * m[13] + 
             m[12] * m[5] * m[11] - 
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] + 
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] - 
               m[8]  * m[6] * m[13] - 
               m[12] * m[5] * m[10] + 
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] + 
              m[1]  * m[11] * m[14] + 
              m[9]  * m[2] * m[15] - 
              m[9]  * m[3] * m[14] - 
              m[13] * m[2] * m[11] + 
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] - 
             m[0]  * m[11] * m[14] - 
             m[8]  * m[2] * m[15] + 
             m[8]  * m[3] * m[14] + 
             m[12] * m[2] * m[11] - 
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] + 
              m[0]  * m[11] * m[13] + 
              m[8]  * m[1] * m[15] - 
              m[8]  * m[3] * m[13] - 
              m[12] * m[1] * m[11] + 
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] - 
              m[0]  * m[10] * m[13] - 
              m[8]  * m[1] * m[14] + 
              m[8]  * m[2] * m[13] + 
              m[12] * m[1] * m[10] - 
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] - 
             m[1]  * m[7] * m[14] - 
             m[5]  * m[2] * m[15] + 
             m[5]  * m[3] * m[14] + 
             m[13] * m[2] * m[7] - 
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] + 
              m[0]  * m[7] * m[14] + 
              m[4]  * m[2] * m[15] - 
              m[4]  * m[3] * m[14] - 
              m[12] * m[2] * m[7] + 
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] - 
              m[0]  * m[7] * m[13] - 
              m[4]  * m[1] * m[15] + 
              m[4]  * m[3] * m[13] + 
              m[12] * m[1] * m[7] - 
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] + 
               m[0]  * m[6] * m[13] + 
               m[4]  * m[1] * m[14] - 
               m[4]  * m[2] * m[13] - 
               m[12] * m[1] * m[6] + 
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + 
              m[1] * m[7] * m[10] + 
              m[5] * m[2] * m[11] - 
              m[5] * m[3] * m[10] - 
              m[9] * m[2] * m[7] + 
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - 
             m[0] * m[7] * m[10] - 
             m[4] * m[2] * m[11] + 
             m[4] * m[3] * m[10] + 
             m[8] * m[2] * m[7] - 
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + 
               m[0] * m[7] * m[9] + 
               m[4] * m[1] * m[11] - 
               m[4] * m[3] * m[9] - 
               m[8] * m[1] * m[7] + 
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - 
              m[0] * m[6] * m[9] - 
              m[4] * m[1] * m[10] + 
              m[4] * m[2] * m[9] + 
              m[8] * m[1] * m[6] - 
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    
    if (det !=0 ){
    det = 1.0 / det;

    for (i = 0; i < 16; i++){
        invOut[i] = inv[i] * det;
    }
    bhat_0 = invOut[0] * x1y_sum + invOut[1] * x2y_sum + invOut[2] * x3y_sum + invOut[3] * x4y_sum;
    bhat_1 = invOut[4] * x1y_sum + invOut[5] * x2y_sum + invOut[6] * x3y_sum + invOut[7] * x4y_sum;
    bhat_2 = invOut[8] * x1y_sum + invOut[9] * x2y_sum + invOut[10] * x3y_sum + invOut[11] * x4y_sum;
    bhat_3 = invOut[12] * x1y_sum + invOut[13] * x2y_sum + invOut[14] * x3y_sum + invOut[15] * x4y_sum;
    for (i = 0; i < n; i++) {
        error2 += (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) * (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) / (n - 4); 
    }
    var3 = error2 * invOut[15];   
    } else {
    bhat_3 = 0;
    var3 = 0;
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
    *risk = 4 * twt * max_y * max_y - alpha * twt * bhat_3 * bhat_3 + (1 - alpha) * (1 + train_to_est_ratio) * twt * (var3);
    //*risk = 4 * twt * max_y * max_y - alpha * twt * effect * effect + (1 - alpha) * (1 + train_to_est_ratio) * twt * (numerator / denominator);
    //if(n * xy_sum - x_sum * y_sum < 0.6 * n * n){
    //    effect = temp1 / ttreat - temp0 / (twt - ttreat);  
    //    *value = effect;
    //    tr_var = tr_sqr_sum / ttreat - temp1 * temp1 / (ttreat * ttreat);
    //    con_var = con_sqr_sum / (twt - ttreat) - temp0 * temp0 / ((twt - ttreat) * (twt - ttreat));
    //    *risk = 4 * twt * max_y * max_y - alpha * twt * effect * effect + 
    //    (1 - alpha) * (1 + train_to_est_ratio) * twt * (tr_var /ttreat  + con_var / (twt - ttreat));
    //}
            
}

void
CTD(int n, double *y[], double *x, int nclass,
    int edge, double *improve, double *split, int *csplit,
    double myrisk, double *wt, double *treatment, double *IV, int minsize,
    double alpha, int bucketnum, int bucketMax, double train_to_est_ratio)
{
    int i, j;
    double temp;
    double left_sum, right_sum;
    double left_tr_sum, right_tr_sum;
    double left_tr_sqr_sum, right_tr_sqr_sum;
    double left_sqr_sum, right_sqr_sum;
    double tr_var, con_var;
    double left_tr_var, left_con_var, right_tr_var, right_con_var;
    double left_tr, right_tr;
    double left_wt, right_wt;
    int left_n, right_n;
    double best;
    int direction = LEFT;
    int where = 0;
    double node_effect, left_effect, right_effect;
    double left_temp, right_temp;
    int min_node_size = minsize;
    int bucketTmp;
    double trsum = 0.;
    int Numbuckets;
    
    double *cum_wt, *tmp_wt, *fake_x;
    double tr_wt_sum, con_wt_sum, con_cum_wt, tr_cum_wt;
    
    // for overlap:
    double tr_min, tr_max, con_min, con_max;
    double left_bd, right_bd;
    double cut_point;
    
    right_wt = 0;
    right_tr = 0;
    right_sum = 0;
    right_tr_sum = 0;
    right_sqr_sum = 0;
    right_tr_sqr_sum = 0;
    right_n = n;
    
    double right_xz_sum = 0., right_xy_sum = 0., right_x_sum = 0., right_y_sum = 0., right_z_sum = 0.;
    double left_xz_sum = 0., left_xy_sum = 0., left_x_sum = 0., left_y_sum = 0., left_z_sum = 0.;
    double right_yz_sum = 0., right_xx_sum = 0., right_yy_sum = 0., right_zz_sum = 0.;
    double left_yz_sum = 0., left_xx_sum = 0., left_yy_sum = 0., left_zz_sum = 0.;
    double alpha_1 = 0., alpha_0 = 0., beta_1 = 0., beta_0 = 0.;
    double numerator, denominator;
    double right_x1x1_sum = 0., right_x1x2_sum = 0., right_x1x3_sum = 0., right_x1x4_sum = 0., right_x2x1_sum = 0., right_x2x2_sum = 0., right_x2x3_sum = 0., right_x2x4_sum = 0., right_x3x1_sum = 0., right_x3x2_sum = 0., right_x3x3_sum = 0., right_x3x4_sum = 0., right_x4x1_sum = 0., right_x4x2_sum = 0., right_x4x3_sum = 0., right_x4x4_sum = 0.;
    double right_x1y_sum = 0., right_x2y_sum = 0., right_x3y_sum = 0., right_x4y_sum = 0.;
    double left_x1x1_sum = 0., left_x1x2_sum = 0., left_x1x3_sum = 0., left_x1x4_sum = 0., left_x2x1_sum = 0., left_x2x2_sum = 0., left_x2x3_sum = 0., left_x2x4_sum = 0., left_x3x1_sum = 0., left_x3x2_sum = 0., left_x3x3_sum = 0., left_x3x4_sum = 0., left_x4x1_sum = 0., left_x4x2_sum = 0., left_x4x3_sum = 0., left_x4x4_sum = 0.;
    double left_x1y_sum = 0., left_x2y_sum = 0., left_x3y_sum = 0., left_x4y_sum = 0.;
    float m[16], inv[16], invOut[16];
    double det;
    double bhat_0 = 0., bhat_1 = 0., bhat_2 = 0., bhat_3 = 0.;
    double error2 = 0., var3 = 0.;
    for (i = 0; i < n; i++) {
        right_wt += wt[i];
        right_tr += wt[i] * treatment[i];
        right_sum += *y[i] * wt[i];
        right_tr_sum += *y[i] * wt[i] * treatment[i];
        right_sqr_sum += (*y[i]) * (*y[i]) * wt[i];
        right_tr_sqr_sum += (*y[i]) * (*y[i]) * wt[i] * treatment[i];
        trsum += treatment[i];
        right_xz_sum += *y[i] * IV[i];
        right_xy_sum += treatment[i] * IV[i];
        right_x_sum += IV[i];
        right_y_sum += treatment[i];
        right_z_sum += *y[i];
        right_yz_sum += *y[i] * treatment[i];
        right_xx_sum += IV[i] * IV[i];
        right_yy_sum += treatment[i] * treatment[i];
        right_zz_sum += *y[i] * *y[i];
	right_x1x1_sum += 1 * 1;
        right_x1x2_sum += 1 * treatment[i];
        right_x1x3_sum += 1 * IV[i];   
        right_x1x4_sum += 1 * IV[i] * treatment[i]; 
        right_x2x1_sum += treatment[i] * 1;
        right_x2x2_sum += treatment[i] * treatment[i];
        right_x2x3_sum += treatment[i] * IV[i];   
        right_x2x4_sum += treatment[i] * IV[i] * treatment[i]; 
        right_x3x1_sum += IV[i] * 1;
        right_x3x2_sum += IV[i] * treatment[i];
        right_x3x3_sum += IV[i] * IV[i];   
        right_x3x4_sum += IV[i] * IV[i] * treatment[i];  
        right_x4x1_sum += IV[i] * treatment[i] * 1; 
        right_x4x2_sum += IV[i] * treatment[i] * treatment[i];
        right_x4x3_sum += IV[i] * treatment[i] * IV[i];   
        right_x4x4_sum += IV[i] * treatment[i] * IV[i] * treatment[i];  
        right_x1y_sum += *y[i];
        right_x2y_sum += *y[i] * treatment[i];
        right_x3y_sum += *y[i] * IV[i];  
        right_x4y_sum += *y[i] * IV[i] * treatment[i];  
    }
    
    //finding determinant
    m[0] = right_x1x1_sum;
    m[1] = right_x1x2_sum;
    m[2] = right_x1x3_sum;
    m[3] = right_x1x4_sum;
    m[4] = right_x2x1_sum;
    m[5] = right_x2x2_sum;
    m[6] = right_x2x3_sum;
    m[7] = right_x2x4_sum;
    m[8] = right_x3x1_sum;
    m[9] = right_x3x2_sum;
    m[10] = right_x3x3_sum;
    m[11] = right_x3x4_sum;
    m[12] = right_x4x1_sum;
    m[13] = right_x4x2_sum;
    m[14] = right_x4x3_sum;     
    m[15] = right_x4x4_sum;   
    inv[0] = m[5]  * m[10] * m[15] - 
             m[5]  * m[11] * m[14] - 
             m[9]  * m[6]  * m[15] + 
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] - 
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] + 
              m[4]  * m[11] * m[14] + 
              m[8]  * m[6]  * m[15] - 
              m[8]  * m[7]  * m[14] - 
              m[12] * m[6]  * m[11] + 
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] - 
             m[4]  * m[11] * m[13] - 
             m[8]  * m[5] * m[15] + 
             m[8]  * m[7] * m[13] + 
             m[12] * m[5] * m[11] - 
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] + 
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] - 
               m[8]  * m[6] * m[13] - 
               m[12] * m[5] * m[10] + 
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] + 
              m[1]  * m[11] * m[14] + 
              m[9]  * m[2] * m[15] - 
              m[9]  * m[3] * m[14] - 
              m[13] * m[2] * m[11] + 
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] - 
             m[0]  * m[11] * m[14] - 
             m[8]  * m[2] * m[15] + 
             m[8]  * m[3] * m[14] + 
             m[12] * m[2] * m[11] - 
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] + 
              m[0]  * m[11] * m[13] + 
              m[8]  * m[1] * m[15] - 
              m[8]  * m[3] * m[13] - 
              m[12] * m[1] * m[11] + 
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] - 
              m[0]  * m[10] * m[13] - 
              m[8]  * m[1] * m[14] + 
              m[8]  * m[2] * m[13] + 
              m[12] * m[1] * m[10] - 
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] - 
             m[1]  * m[7] * m[14] - 
             m[5]  * m[2] * m[15] + 
             m[5]  * m[3] * m[14] + 
             m[13] * m[2] * m[7] - 
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] + 
              m[0]  * m[7] * m[14] + 
              m[4]  * m[2] * m[15] - 
              m[4]  * m[3] * m[14] - 
              m[12] * m[2] * m[7] + 
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] - 
              m[0]  * m[7] * m[13] - 
              m[4]  * m[1] * m[15] + 
              m[4]  * m[3] * m[13] + 
              m[12] * m[1] * m[7] - 
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] + 
               m[0]  * m[6] * m[13] + 
               m[4]  * m[1] * m[14] - 
               m[4]  * m[2] * m[13] - 
               m[12] * m[1] * m[6] + 
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + 
              m[1] * m[7] * m[10] + 
              m[5] * m[2] * m[11] - 
              m[5] * m[3] * m[10] - 
              m[9] * m[2] * m[7] + 
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - 
             m[0] * m[7] * m[10] - 
             m[4] * m[2] * m[11] + 
             m[4] * m[3] * m[10] + 
             m[8] * m[2] * m[7] - 
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + 
               m[0] * m[7] * m[9] + 
               m[4] * m[1] * m[11] - 
               m[4] * m[3] * m[9] - 
               m[8] * m[1] * m[7] + 
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - 
              m[0] * m[6] * m[9] - 
              m[4] * m[1] * m[10] + 
              m[4] * m[2] * m[9] + 
              m[8] * m[1] * m[6] - 
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det != 0){	
    det = 1.0 / det;

    for (i = 0; i < 16; i++){
        invOut[i] = inv[i] * det;
    }
    bhat_0 = invOut[0] * right_x1y_sum + invOut[1] * right_x2y_sum + invOut[2] * right_x3y_sum + invOut[3] * right_x4y_sum;
    bhat_1 = invOut[4] * right_x1y_sum + invOut[5] * right_x2y_sum + invOut[6] * right_x3y_sum + invOut[7] * right_x4y_sum;
    bhat_2 = invOut[8] * right_x1y_sum + invOut[9] * right_x2y_sum + invOut[10] * right_x3y_sum + invOut[11] * right_x4y_sum;
    bhat_3 = invOut[12] * right_x1y_sum + invOut[13] * right_x2y_sum + invOut[14] * right_x3y_sum + invOut[15] * right_x4y_sum;
    for (i = 0; i < n; i++) {
        error2 += (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) * (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) / (n - 4); 
    }
    var3 = error2 * invOut[15];   
    } else {
    bhat_3 = 0;
    var3 = 0; 
    }
	
    alpha_1 = (right_n * right_xz_sum - right_x_sum * right_z_sum) / (right_n * right_xy_sum - right_x_sum * right_y_sum);
    alpha_0 = (right_z_sum - alpha_1 * right_y_sum) / right_n;
    beta_1 = (right_n * right_xy_sum - right_x_sum * right_y_sum) / (right_n * right_xx_sum - right_x_sum * right_x_sum);
    beta_0 = (right_y_sum - beta_1 * right_x_sum) / right_n;
    temp = alpha_1;
    numerator = right_zz_sum + right_n * alpha_0 * alpha_0 + alpha_1 * alpha_1 * right_yy_sum - 2 * alpha_0 * right_z_sum - 2 * alpha_1 * right_yz_sum + 2 * alpha_0 * alpha_1 * right_y_sum;
    denominator = right_n * beta_0 * beta_0 + beta_1 * beta_1 * right_xx_sum + right_y_sum * right_y_sum / right_n + 2 * beta_0 * beta_1 * right_x_sum - 2 * beta_0 * right_y_sum - 2 * beta_1 * right_x_sum * right_y_sum / right_n;
    node_effect = alpha * bhat_3 * bhat_3 * right_wt - (1 - alpha) * (1 + train_to_est_ratio)
    * right_wt * (var3);
    
    //if (right_n * right_xy_sum - right_x_sum * right_y_sum < 0.6 * right_n * right_n){
    //temp = right_tr_sum / right_tr - (right_sum - right_tr_sum) / (right_wt - right_tr);
    //tr_var = right_tr_sqr_sum / right_tr - right_tr_sum * right_tr_sum / (right_tr * right_tr);
    //con_var = (right_sqr_sum - right_tr_sqr_sum) / (right_wt - right_tr)
    //	- (right_sum - right_tr_sum) * (right_sum - right_tr_sum)
    // 	/ ((right_wt - right_tr) * (right_wt - right_tr));
    //node_effect = alpha * temp * temp * right_wt - (1 - alpha) * (1 + train_to_est_ratio)
    //	* right_wt * (tr_var / right_tr  + con_var / (right_wt - right_tr));}
    
    if (nclass == 0) {
        /* continuous predictor */
        cum_wt = (double *) ALLOC(n, sizeof(double));
        tmp_wt = (double *) ALLOC(n, sizeof(double));
        fake_x = (double *) ALLOC(n, sizeof(double));
        
        tr_wt_sum = 0.;
        con_wt_sum = 0.;
        con_cum_wt = 0.;
        tr_cum_wt = 0.;
        
        // find the abs max and min of x:
        double max_abs_tmp = fabs(x[0]);
        for (i = 0; i < n; i++) {
            if (max_abs_tmp < fabs(x[i])) {
                max_abs_tmp = fabs(x[i]);
            }
        }
        
        // set tr_min, con_min, tr_max, con_max to a large/small value
        tr_min = max_abs_tmp;
        tr_max = -max_abs_tmp;
        con_min = max_abs_tmp;
        con_max = -max_abs_tmp;
        
        for (i = 0; i < n; i++) {
            if (treatment[i] == 0) {
                con_wt_sum += wt[i];
                if (con_min > x[i]) {
                    con_min = x[i];
                }
                if (con_max < x[i]) {
                    con_max = x[i];
                }
            } else {
                tr_wt_sum += wt[i];
                if (tr_min > x[i]) {
                    tr_min = x[i];
                }
                if (tr_max < x[i]) {
                    tr_max = x[i];
                }
            }
            cum_wt[i] = 0.;
            tmp_wt[i] = 0.;
            fake_x[i] = 0.;
        }
        
        // compute the left bound and right bound
        left_bd = max(tr_min, con_min);
        right_bd = min(tr_max, con_max);
        
        bucketTmp = min(round(trsum / (double)bucketnum), round(((double)n - trsum) / (double)bucketnum));
        Numbuckets = max(minsize, min(bucketTmp, bucketMax));
        
        for (i = 0; i < n; i++) {
            if (treatment[i] == 0) {
                tmp_wt[i] = wt[i] / con_wt_sum;
                con_cum_wt += tmp_wt[i];
                cum_wt[i] = con_cum_wt;
                fake_x[i] = (int)floor(Numbuckets * cum_wt[i]);
            } else {
                tmp_wt[i] = wt[i] / tr_wt_sum;
                tr_cum_wt += tmp_wt[i];
                cum_wt[i] = tr_cum_wt;
                fake_x[i] = (int)floor(Numbuckets * cum_wt[i]);
            }
        }
        
        n_bucket = (int *) ALLOC(Numbuckets + 1,  sizeof(int));
        n_tr_bucket = (int *) ALLOC(Numbuckets + 1, sizeof(int));
        n_con_bucket = (int *) ALLOC(Numbuckets + 1, sizeof(int));
        wts_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        trs_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        wtsums_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        trsums_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        wtsqrsums_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        trsqrsums_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        tr_end_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        con_end_bucket = (double *) ALLOC (Numbuckets + 1, sizeof(double));
        
        xz_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        xy_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        y_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        z_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        yz_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        xx_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        yy_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        zz_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        
	x1x1_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x1x2_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x1x3_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x1x4_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x2x1_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x2x2_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x2x3_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x2x4_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x3x1_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x3x2_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x3x3_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x3x4_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x4x1_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x4x2_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x4x3_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x4x4_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x1y_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x2y_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double)); 
        x3y_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));
        x4y_sum_bucket = (double *) ALLOC(Numbuckets + 1, sizeof(double));    
	    
        for (j = 0; j < Numbuckets + 1; j++) {
            n_bucket[j] = 0;
            n_tr_bucket[j] = 0;
            n_con_bucket[j] = 0;
            wts_bucket[j] = 0.;
            trs_bucket[j] = 0.;
            wtsums_bucket[j] = 0.;
            trsums_bucket[j] = 0.;
            wtsqrsums_bucket[j] = 0.;
            trsqrsums_bucket[j] = 0.;
            
            xz_sum_bucket[j] = 0.;
            xy_sum_bucket[j] = 0.;
            x_sum_bucket[j] = 0.;
            y_sum_bucket[j] = 0.;
            z_sum_bucket[j] = 0.;
            yz_sum_bucket[j] = 0.;
            xx_sum_bucket[j] = 0.;
            yy_sum_bucket[j] = 0.;
            zz_sum_bucket[j] = 0.;
	x1x1_sum_bucket[j] = 0.;
        x1x2_sum_bucket[j] = 0.;
        x1x3_sum_bucket[j] = 0.;
        x1x4_sum_bucket[j] = 0.;
        x2x1_sum_bucket[j] = 0.;
        x2x2_sum_bucket[j] = 0.;
        x2x3_sum_bucket[j] = 0.;
        x2x4_sum_bucket[j] = 0.;
        x3x1_sum_bucket[j] = 0.;
        x3x2_sum_bucket[j] = 0.;
        x3x3_sum_bucket[j] = 0.;
        x3x4_sum_bucket[j] = 0.;
        x4x1_sum_bucket[j] = 0.;
        x4x2_sum_bucket[j] = 0.;
        x4x3_sum_bucket[j] = 0.;
        x4x4_sum_bucket[j] = 0.;
        x1y_sum_bucket[j] = 0.;
        x2y_sum_bucket[j] = 0.; 
        x3y_sum_bucket[j] = 0.;
        x4y_sum_bucket[j] = 0.;  
        }
        
        for (i = 0; i < n; i++) {
            j = fake_x[i];
            n_bucket[j]++;
            wts_bucket[j] += wt[i];
            trs_bucket[j] += wt[i] * treatment[i];
            wtsums_bucket[j] += *y[i] * wt[i];
            trsums_bucket[j] += *y[i] * wt[i] * treatment[i];
            wtsqrsums_bucket[j] += (*y[i]) * (*y[i]) * wt[i];
            trsqrsums_bucket[j] += (*y[i]) * (*y[i]) * wt[i] * treatment[i];
            
            xz_sum_bucket[j] += *y[i] * IV[i];
            xy_sum_bucket[j] += treatment[i] * IV[i];
            x_sum_bucket[j] += IV[i];
            y_sum_bucket[j] += treatment[i];
            z_sum_bucket[j] += *y[i];
            yz_sum_bucket[j] += *y[i] * treatment[i];
            xx_sum_bucket[j] += IV[i] * IV[i];
            yy_sum_bucket[j] += treatment[i] * treatment[i];
            zz_sum_bucket[j] += *y[i] * *y[i];
	x1x1_sum_bucket[j] += 1 * 1;
        x1x2_sum_bucket[j] += 1 * treatment[i];
        x1x3_sum_bucket[j] += 1 * IV[i];   
        x1x4_sum_bucket[j] += 1 * IV[i] * treatment[i]; 
        x2x1_sum_bucket[j] += treatment[i] * 1;
        x2x2_sum_bucket[j] += treatment[i] * treatment[i];
        x2x3_sum_bucket[j] += treatment[i] * IV[i];   
        x2x4_sum_bucket[j] += treatment[i] * IV[i] * treatment[i]; 
        x3x1_sum_bucket[j] += IV[i] * 1;
        x3x2_sum_bucket[j] += IV[i] * treatment[i];
        x3x3_sum_bucket[j] += IV[i] * IV[i];   
        x3x4_sum_bucket[j] += IV[i] * IV[i] * treatment[i];  
        x4x1_sum_bucket[j] += IV[i] * treatment[i] * 1; 
        x4x2_sum_bucket[j] += IV[i] * treatment[i] * treatment[i];
        x4x3_sum_bucket[j] += IV[i] * treatment[i] * IV[i];   
        x4x4_sum_bucket[j] += IV[i] * treatment[i] * IV[i] * treatment[i];  
        x1y_sum_bucket[j] += *y[i];
        x2y_sum_bucket[j] += *y[i] * treatment[i];
        x3y_sum_bucket[j] += *y[i] * IV[i];  
        x4y_sum_bucket[j] += *y[i] * IV[i] * treatment[i];  
            
            if (treatment[i] == 1) {
                tr_end_bucket[j] = x[i];
            } else {
                con_end_bucket[j] = x[i];
            }
        }
        
        left_wt = 0;
        left_tr = 0;
        left_n = 0;
        left_sum = 0;
        left_tr_sum = 0;
        left_sqr_sum = 0;
        left_tr_sqr_sum = 0;
        left_temp = 0.;
        right_temp = 0.;
        
        best = 0;
        
        for (j = 0; j < Numbuckets; j++) {
            left_n += n_bucket[j];
            right_n -= n_bucket[j];
            left_wt += wts_bucket[j];
            right_wt -= wts_bucket[j];
            left_tr += trs_bucket[j];
            right_tr -= trs_bucket[j];
            
            left_sum += wtsums_bucket[j];
            right_sum -= wtsums_bucket[j];
            
            left_tr_sum += trsums_bucket[j];
            right_tr_sum -= trsums_bucket[j];
            
            left_sqr_sum += wtsqrsums_bucket[j];
            right_sqr_sum -= wtsqrsums_bucket[j];
            
            left_tr_sqr_sum += trsqrsums_bucket[j];
            right_tr_sqr_sum -= trsqrsums_bucket[j];
            
            left_xz_sum += xz_sum_bucket[j];
            right_xz_sum -= xz_sum_bucket[j];
            left_xy_sum += xy_sum_bucket[j];
            right_xy_sum -= xy_sum_bucket[j];
            left_x_sum += x_sum_bucket[j];
            right_x_sum -= x_sum_bucket[j];
            left_y_sum += y_sum_bucket[j];
            right_y_sum -= y_sum_bucket[j];
            left_z_sum += z_sum_bucket[j];
            right_z_sum -= z_sum_bucket[j];
            left_yz_sum += yz_sum_bucket[j];
            right_yz_sum -= yz_sum_bucket[j];
            left_xx_sum += xx_sum_bucket[j];
            right_xx_sum -= xx_sum_bucket[j];
            left_yy_sum += yy_sum_bucket[j];
            right_yy_sum -= yy_sum_bucket[j];
            left_zz_sum += zz_sum_bucket[j];
            right_zz_sum -= zz_sum_bucket[j];
            
	    left_x1x1_sum += x1x1_sum_bucket[j];
            right_x1x1_sum -= x1x1_sum_bucket[j];
            left_x1x2_sum += x1x2_sum_bucket[j];
            right_x1x2_sum -= x1x2_sum_bucket[j];
            left_x1x3_sum += x1x3_sum_bucket[j];  
            right_x1x3_sum -= x1x3_sum_bucket[j];  
            left_x1x4_sum += x1x4_sum_bucket[j];   
            right_x1x4_sum -= x1x4_sum_bucket[j]; 
            left_x2x1_sum += x2x1_sum_bucket[j];
            right_x2x1_sum -= x2x1_sum_bucket[j];
            left_x2x2_sum += x2x2_sum_bucket[j];
            right_x2x2_sum -= x2x2_sum_bucket[j];
            left_x2x3_sum += x2x3_sum_bucket[j];    
            right_x2x3_sum -= x2x3_sum_bucket[j]; 
            left_x2x4_sum += x2x4_sum_bucket[j]; 
            right_x2x4_sum -= x2x4_sum_bucket[j]; 
            left_x3x1_sum += x3x1_sum_bucket[j];
            right_x3x1_sum -= x3x1_sum_bucket[j];
            left_x3x2_sum += x3x2_sum_bucket[j];
            right_x3x2_sum -= x3x2_sum_bucket[j];
            left_x3x3_sum += x3x3_sum_bucket[j]; 
            right_x3x3_sum -= x3x3_sum_bucket[j]; 
            left_x3x4_sum += x3x4_sum_bucket[j];
            right_x3x4_sum -= x3x4_sum_bucket[j];  
            left_x4x1_sum += x4x1_sum_bucket[j]; 
            right_x4x1_sum -= x4x1_sum_bucket[j];
            left_x4x2_sum += x4x2_sum_bucket[j];
            right_x4x2_sum -= x4x2_sum_bucket[j];
            left_x4x3_sum += x4x3_sum_bucket[j];
            right_x4x3_sum -= x4x3_sum_bucket[j];
            left_x4x4_sum += x4x4_sum_bucket[j]; 
            right_x4x4_sum -= x4x4_sum_bucket[j];  
            left_x1y_sum += x1y_sum_bucket[j];
            right_x1y_sum -= x1y_sum_bucket[j];
            left_x2y_sum += x2y_sum_bucket[j];
            right_x2y_sum -= x2y_sum_bucket[j];
            left_x3y_sum += x3y_sum_bucket[j];
            right_x3y_sum -= x3y_sum_bucket[j];
            left_x4y_sum += x4y_sum_bucket[j];
            right_x4y_sum -= x4y_sum_bucket[j];
            
            cut_point = (tr_end_bucket[j] + con_end_bucket[j]) / 2.0;
            if (left_n >= edge && right_n >= edge &&
                (int) left_tr >= min_node_size &&
                (int) left_wt - (int) left_tr >= min_node_size &&
                (int) right_tr >= min_node_size &&
                (int) right_wt - (int) right_tr >= min_node_size &&
                cut_point < right_bd && cut_point > left_bd) {
              
    m[0] = left_x1x1_sum;
    m[1] = left_x1x2_sum;
    m[2] = left_x1x3_sum;
    m[3] = left_x1x4_sum;
    m[4] = left_x2x1_sum;
    m[5] = left_x2x2_sum;
    m[6] = left_x2x3_sum;
    m[7] = left_x2x4_sum;
    m[8] = left_x3x1_sum;
    m[9] = left_x3x2_sum;
    m[10] = left_x3x3_sum;
    m[11] = left_x3x4_sum;
    m[12] = left_x4x1_sum;
    m[13] = left_x4x2_sum;
    m[14] = left_x4x3_sum;     
    m[15] = left_x4x4_sum;   
    inv[0] = m[5]  * m[10] * m[15] - 
             m[5]  * m[11] * m[14] - 
             m[9]  * m[6]  * m[15] + 
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] - 
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] + 
              m[4]  * m[11] * m[14] + 
              m[8]  * m[6]  * m[15] - 
              m[8]  * m[7]  * m[14] - 
              m[12] * m[6]  * m[11] + 
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] - 
             m[4]  * m[11] * m[13] - 
             m[8]  * m[5] * m[15] + 
             m[8]  * m[7] * m[13] + 
             m[12] * m[5] * m[11] - 
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] + 
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] - 
               m[8]  * m[6] * m[13] - 
               m[12] * m[5] * m[10] + 
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] + 
              m[1]  * m[11] * m[14] + 
              m[9]  * m[2] * m[15] - 
              m[9]  * m[3] * m[14] - 
              m[13] * m[2] * m[11] + 
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] - 
             m[0]  * m[11] * m[14] - 
             m[8]  * m[2] * m[15] + 
             m[8]  * m[3] * m[14] + 
             m[12] * m[2] * m[11] - 
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] + 
              m[0]  * m[11] * m[13] + 
              m[8]  * m[1] * m[15] - 
              m[8]  * m[3] * m[13] - 
              m[12] * m[1] * m[11] + 
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] - 
              m[0]  * m[10] * m[13] - 
              m[8]  * m[1] * m[14] + 
              m[8]  * m[2] * m[13] + 
              m[12] * m[1] * m[10] - 
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] - 
             m[1]  * m[7] * m[14] - 
             m[5]  * m[2] * m[15] + 
             m[5]  * m[3] * m[14] + 
             m[13] * m[2] * m[7] - 
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] + 
              m[0]  * m[7] * m[14] + 
              m[4]  * m[2] * m[15] - 
              m[4]  * m[3] * m[14] - 
              m[12] * m[2] * m[7] + 
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] - 
              m[0]  * m[7] * m[13] - 
              m[4]  * m[1] * m[15] + 
              m[4]  * m[3] * m[13] + 
              m[12] * m[1] * m[7] - 
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] + 
               m[0]  * m[6] * m[13] + 
               m[4]  * m[1] * m[14] - 
               m[4]  * m[2] * m[13] - 
               m[12] * m[1] * m[6] + 
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + 
              m[1] * m[7] * m[10] + 
              m[5] * m[2] * m[11] - 
              m[5] * m[3] * m[10] - 
              m[9] * m[2] * m[7] + 
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - 
             m[0] * m[7] * m[10] - 
             m[4] * m[2] * m[11] + 
             m[4] * m[3] * m[10] + 
             m[8] * m[2] * m[7] - 
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + 
               m[0] * m[7] * m[9] + 
               m[4] * m[1] * m[11] - 
               m[4] * m[3] * m[9] - 
               m[8] * m[1] * m[7] + 
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - 
              m[0] * m[6] * m[9] - 
              m[4] * m[1] * m[10] + 
              m[4] * m[2] * m[9] + 
              m[8] * m[1] * m[6] - 
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det != 0){	    
    det = 1.0 / det;

    for (i = 0; i < 16; i++){
        invOut[i] = inv[i] * det;
    }
    bhat_0 = invOut[0] * left_x1y_sum + invOut[1] * left_x2y_sum + invOut[2] * left_x3y_sum + invOut[3] * left_x4y_sum;
    bhat_1 = invOut[4] * left_x1y_sum + invOut[5] * left_x2y_sum + invOut[6] * left_x3y_sum + invOut[7] * left_x4y_sum;
    bhat_2 = invOut[8] * left_x1y_sum + invOut[9] * left_x2y_sum + invOut[10] * left_x3y_sum + invOut[11] * left_x4y_sum;
    bhat_3 = invOut[12] * left_x1y_sum + invOut[13] * left_x2y_sum + invOut[14] * left_x3y_sum + invOut[15] * left_x4y_sum;
    for (i = 0; i < left_n; i++) {
        error2 += (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) * (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) / (n - 4); 
    }
    var3 = error2 * invOut[15];  
    } else {
    bhat_3 = 0;
    var3 = 0;
    }   
                alpha_1 = (left_n * left_xz_sum - left_x_sum * left_z_sum) / (left_n * left_xy_sum - left_x_sum * left_y_sum);
                alpha_0 = (left_z_sum - alpha_1 * left_y_sum) / left_n;
                beta_1 = (left_n * left_xy_sum - left_x_sum * left_y_sum) / (left_n * left_xx_sum - left_x_sum * left_x_sum);
                beta_0 = (left_y_sum - beta_1 * left_x_sum) / left_n;
                left_temp = alpha_1;
                numerator = left_zz_sum + left_n * alpha_0 * alpha_0 + alpha_1 * alpha_1 * left_yy_sum - 2 * alpha_0 * left_z_sum - 2 * alpha_1 * left_yz_sum + 2 * alpha_0 * alpha_1 * left_y_sum;
                denominator = left_n * beta_0 * beta_0 + beta_1 * beta_1 * left_xx_sum + left_y_sum * left_y_sum / left_n + 2 * beta_0 * beta_1 * left_x_sum - 2 * beta_0 * left_y_sum - 2 * beta_1 * left_x_sum * left_y_sum / left_n;
                //left_effect = alpha * left_temp * left_temp * left_wt - (1 - alpha) * (1 + train_to_est_ratio)
                //* left_wt * (numerator / denominator);
		left_effect = alpha * bhat_3 * bhat_3 * left_wt - (1 - alpha) * (1 + train_to_est_ratio)
                * left_wt * (var3);
                
              //  if(left_n * left_xy_sum - left_x_sum * left_y_sum < 0.6 * left_n * left_n){
              //      left_temp = left_tr_sum / left_tr - (left_sum - left_tr_sum) / (left_wt - left_tr);
              //      left_tr_var = left_tr_sqr_sum / left_tr -
              //      left_tr_sum  * left_tr_sum / (left_tr * left_tr);
              //      left_con_var = (left_sqr_sum - left_tr_sqr_sum) / (left_wt - left_tr)
              //      - (left_sum - left_tr_sum) * (left_sum - left_tr_sum)
              //      / ((left_wt - left_tr) * (left_wt - left_tr));
              //      left_effect = alpha * left_temp * left_temp * left_wt
              //      - (1 - alpha) * (1 + train_to_est_ratio) * left_wt
              //      * (left_tr_var / left_tr + left_con_var / (left_wt - left_tr));}
                
    m[0] = right_x1x1_sum;
    m[1] = right_x1x2_sum;
    m[2] = right_x1x3_sum;
    m[3] = right_x1x4_sum;
    m[4] = right_x2x1_sum;
    m[5] = right_x2x2_sum;
    m[6] = right_x2x3_sum;
    m[7] = right_x2x4_sum;
    m[8] = right_x3x1_sum;
    m[9] = right_x3x2_sum;
    m[10] = right_x3x3_sum;
    m[11] = right_x3x4_sum;
    m[12] = right_x4x1_sum;
    m[13] = right_x4x2_sum;
    m[14] = right_x4x3_sum;     
    m[15] = right_x4x4_sum;   
    inv[0] = m[5]  * m[10] * m[15] - 
             m[5]  * m[11] * m[14] - 
             m[9]  * m[6]  * m[15] + 
             m[9]  * m[7]  * m[14] +
             m[13] * m[6]  * m[11] - 
             m[13] * m[7]  * m[10];

    inv[4] = -m[4]  * m[10] * m[15] + 
              m[4]  * m[11] * m[14] + 
              m[8]  * m[6]  * m[15] - 
              m[8]  * m[7]  * m[14] - 
              m[12] * m[6]  * m[11] + 
              m[12] * m[7]  * m[10];

    inv[8] = m[4]  * m[9] * m[15] - 
             m[4]  * m[11] * m[13] - 
             m[8]  * m[5] * m[15] + 
             m[8]  * m[7] * m[13] + 
             m[12] * m[5] * m[11] - 
             m[12] * m[7] * m[9];

    inv[12] = -m[4]  * m[9] * m[14] + 
               m[4]  * m[10] * m[13] +
               m[8]  * m[5] * m[14] - 
               m[8]  * m[6] * m[13] - 
               m[12] * m[5] * m[10] + 
               m[12] * m[6] * m[9];

    inv[1] = -m[1]  * m[10] * m[15] + 
              m[1]  * m[11] * m[14] + 
              m[9]  * m[2] * m[15] - 
              m[9]  * m[3] * m[14] - 
              m[13] * m[2] * m[11] + 
              m[13] * m[3] * m[10];

    inv[5] = m[0]  * m[10] * m[15] - 
             m[0]  * m[11] * m[14] - 
             m[8]  * m[2] * m[15] + 
             m[8]  * m[3] * m[14] + 
             m[12] * m[2] * m[11] - 
             m[12] * m[3] * m[10];

    inv[9] = -m[0]  * m[9] * m[15] + 
              m[0]  * m[11] * m[13] + 
              m[8]  * m[1] * m[15] - 
              m[8]  * m[3] * m[13] - 
              m[12] * m[1] * m[11] + 
              m[12] * m[3] * m[9];

    inv[13] = m[0]  * m[9] * m[14] - 
              m[0]  * m[10] * m[13] - 
              m[8]  * m[1] * m[14] + 
              m[8]  * m[2] * m[13] + 
              m[12] * m[1] * m[10] - 
              m[12] * m[2] * m[9];

    inv[2] = m[1]  * m[6] * m[15] - 
             m[1]  * m[7] * m[14] - 
             m[5]  * m[2] * m[15] + 
             m[5]  * m[3] * m[14] + 
             m[13] * m[2] * m[7] - 
             m[13] * m[3] * m[6];

    inv[6] = -m[0]  * m[6] * m[15] + 
              m[0]  * m[7] * m[14] + 
              m[4]  * m[2] * m[15] - 
              m[4]  * m[3] * m[14] - 
              m[12] * m[2] * m[7] + 
              m[12] * m[3] * m[6];

    inv[10] = m[0]  * m[5] * m[15] - 
              m[0]  * m[7] * m[13] - 
              m[4]  * m[1] * m[15] + 
              m[4]  * m[3] * m[13] + 
              m[12] * m[1] * m[7] - 
              m[12] * m[3] * m[5];

    inv[14] = -m[0]  * m[5] * m[14] + 
               m[0]  * m[6] * m[13] + 
               m[4]  * m[1] * m[14] - 
               m[4]  * m[2] * m[13] - 
               m[12] * m[1] * m[6] + 
               m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + 
              m[1] * m[7] * m[10] + 
              m[5] * m[2] * m[11] - 
              m[5] * m[3] * m[10] - 
              m[9] * m[2] * m[7] + 
              m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - 
             m[0] * m[7] * m[10] - 
             m[4] * m[2] * m[11] + 
             m[4] * m[3] * m[10] + 
             m[8] * m[2] * m[7] - 
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + 
               m[0] * m[7] * m[9] + 
               m[4] * m[1] * m[11] - 
               m[4] * m[3] * m[9] - 
               m[8] * m[1] * m[7] + 
               m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - 
              m[0] * m[6] * m[9] - 
              m[4] * m[1] * m[10] + 
              m[4] * m[2] * m[9] + 
              m[8] * m[1] * m[6] - 
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    if (det != 0){
    det = 1.0 / det;

    for (i = 0; i < 16; i++){
        invOut[i] = inv[i] * det;
    }
    bhat_0 = invOut[0] * right_x1y_sum + invOut[1] * right_x2y_sum + invOut[2] * right_x3y_sum + invOut[3] * right_x4y_sum;
    bhat_1 = invOut[4] * right_x1y_sum + invOut[5] * right_x2y_sum + invOut[6] * right_x3y_sum + invOut[7] * right_x4y_sum;
    bhat_2 = invOut[8] * right_x1y_sum + invOut[9] * right_x2y_sum + invOut[10] * right_x3y_sum + invOut[11] * right_x4y_sum;
    bhat_3 = invOut[12] * right_x1y_sum + invOut[13] * right_x2y_sum + invOut[14] * right_x3y_sum + invOut[15] * right_x4y_sum;
    for (i = left_n; i < n; i++) {
        error2 += (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) * (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) / (n - 4); 
    }
    var3 = error2 * invOut[15];   
    } else {
    bhat_3 = 0;
    var3 = 0;
    }
                alpha_1 = (right_n * right_xz_sum - right_x_sum * right_z_sum) / (right_n * right_xy_sum - right_x_sum * right_y_sum);
                alpha_0 = (right_z_sum - alpha_1 * right_y_sum) / right_n;
                beta_1 = (right_n * right_xy_sum - right_x_sum * right_y_sum) / (right_n * right_xx_sum - right_x_sum * right_x_sum);
                beta_0 = (right_y_sum - beta_1 * right_x_sum) / right_n;
                right_temp = alpha_1;
                numerator = right_zz_sum + right_n * alpha_0 * alpha_0 + alpha_1 * alpha_1 * right_yy_sum - 2 * alpha_0 * right_z_sum - 2 * alpha_1 * right_yz_sum + 2 * alpha_0 * alpha_1 * right_y_sum;
                denominator = right_n * beta_0 * beta_0 + beta_1 * beta_1 * right_xx_sum + right_y_sum * right_y_sum / right_n + 2 * beta_0 * beta_1 * right_x_sum - 2 * beta_0 * right_y_sum - 2 * beta_1 * right_x_sum * right_y_sum / right_n;
                //right_effect = alpha * right_temp * right_temp * right_wt - (1 - alpha) * (1 + train_to_est_ratio)
                //* right_wt * (numerator / denominator);
		right_effect = alpha * bhat_3 * bhat_3 * right_wt - (1 - alpha) * (1 + train_to_est_ratio)
                * right_wt * (var3);
                
          //      if(right_n * right_xy_sum - right_x_sum * right_y_sum < 0.6 * right_n * right_n){
          //          right_temp = right_tr_sum / right_tr - (right_sum - right_tr_sum) / (right_wt - right_tr);
          //          right_tr_var = right_tr_sqr_sum / right_tr -
          //          right_tr_sum * right_tr_sum / (right_tr * right_tr);
          //          right_con_var = (right_sqr_sum - right_tr_sqr_sum) / (right_wt - right_tr)
          //          - (right_sum - right_tr_sum) * (right_sum - right_tr_sum)
          //          / ((right_wt - right_tr) * (right_wt - right_tr));
          //          right_effect = alpha * right_temp * right_temp * right_wt
          //          - (1 - alpha) * (1 + train_to_est_ratio) * right_wt *
          //          (right_tr_var / right_tr + right_con_var / (right_wt - right_tr));}
                
                temp = left_effect + right_effect - node_effect;
                if (temp > best) {
                    best = temp;
                    where = j;
                    if (left_temp < right_temp)
                        direction = LEFT;
                    else
                        direction = RIGHT;
                }
            }
        }
        
        *improve = best;
        if (best > 0) {         /* found something */
            csplit[0] = direction;
            *split = (tr_end_bucket[where] + con_end_bucket[where]) / 2.0;
        }
        
    } else {
        /*
         * Categorical predictor
         */
        for (i = 0; i < nclass; i++) {
            countn[i] = 0;
            wts[i] = 0;
            trs[i] = 0;
            sums[i] = 0;
            wtsums[i] = 0;
            trsums[i] = 0;
            wtsqrsums[i] = 0;
            wttrsqrsums[i] = 0;
        }
        
        
        for (i = 0; i < n; i++) {
            j = (int) x[i] - 1;
            countn[j]++;
            wts[j] += wt[i];
            trs[j] += wt[i] * treatment[i];
            sums[j] += *y[i];
            wtsums[j] += *y[i] * wt[i];
            trsums[j] += *y[i] * wt[i] * treatment[i];
            wtsqrsums[j] += (*y[i]) * (*y[i]) * wt[i];
            wttrsqrsums[j] += (*y[i]) * (*y[i]) * wt[i] * treatment[i];
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
        left_sqr_sum = 0;
        left_tr_sqr_sum = 0;
        
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
            
            left_tr_sqr_sum += wttrsqrsums[j];
            right_tr_sqr_sum -= wttrsqrsums[j];
            
            
            if (left_n >= edge && right_n >= edge && 
                (int) left_tr >= min_node_size &&
                (int) left_wt - (int) left_tr >= min_node_size &&
                (int) right_tr >= min_node_size &&
                (int) right_wt - (int) right_tr >= min_node_size) {
                
                left_temp = left_tr_sum / left_tr - (left_sum - left_tr_sum) / (left_wt - left_tr);
                left_tr_var = left_tr_sqr_sum / left_tr - left_tr_sum  * left_tr_sum / (left_tr * left_tr);
                left_con_var = (left_sqr_sum - left_tr_sqr_sum) / (left_wt - left_tr) 
                - (left_sum - left_tr_sum) * (left_sum - left_tr_sum)
                / ((left_wt - left_tr) * (left_wt - left_tr));        
                left_effect = alpha * left_temp * left_temp * left_wt
                - (1 - alpha) * (1 + train_to_est_ratio) * left_wt 
                * (left_tr_var / left_tr + left_con_var / (left_wt - left_tr));
                
                right_temp = right_tr_sum / right_tr - (right_sum - right_tr_sum) / (right_wt - right_tr);
                right_tr_var = right_tr_sqr_sum / right_tr - 
                right_tr_sum * right_tr_sum / (right_tr * right_tr);
                right_con_var = (right_sqr_sum - right_tr_sqr_sum) / (right_wt - right_tr)
                - (right_sum - right_tr_sum) * (right_sum - right_tr_sum) 
                / ((right_wt - right_tr) * (right_wt - right_tr));
		// may need to update below
                right_effect = alpha * right_temp * right_temp * right_wt 
                - (1 - alpha) * (1 + train_to_est_ratio) * right_wt 
                * (right_tr_var / right_tr + right_con_var / (right_wt - right_tr)); 
                
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

double CTDpred(double *y, double wt, double treatment, double *yhat, double propensity) // pass in ct.which
{
	double ystar;
	double temp;

	ystar = y[0] * (treatment - propensity) / (propensity * (1 - propensity));
	temp = ystar - *yhat;
	return temp * temp * wt;
}
