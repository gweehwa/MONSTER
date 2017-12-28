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
static double *xz_sumc, *xy_sumc, *x_sumc, *y_sumc, *z_sumc, *yz_sumc, *xx_sumc, *yy_sumc, *zz_sumc; //declare double for categorical
static double *x1x1_sumc, *x1x2_sumc, *x1x3_sumc, *x1x4_sumc, *x2x1_sumc, *x2x2_sumc, *x2x3_sumc, *x2x4_sumc, 
              *x3x1_sumc, *x3x2_sumc, *x3x3_sumc, *x3x4_sumc, *x4x1_sumc, *x4x2_sumc, *x4x3_sumc, *x4x4_sumc,
              *x1y_sumc, *x2y_sumc, *x3y_sumc, *x4y_sumc;

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
        x_sumc = (double *) ALLOC(9 * maxcat, sizeof(double));
        y_sumc = x_sumc + maxcat;
        z_sumc = y_sumc + maxcat;
        xy_sumc = z_sumc + maxcat;
        xz_sumc = xy_sumc + maxcat;
        yz_sumc = xz_sumc + maxcat;
        xx_sumc = yz_sumc + maxcat;
        yy_sumc = xx_sumc + maxcat;
        zz_sumc = yy_sumc + maxcat;
        x1x1_sumc = (double *) ALLOC(9 * maxcat, sizeof(double));
        x1x2_sumc = x1x1_sumc + maxcat;
        x1x3_sumc = x1x1_sumc + maxcat;
        x1x4_sumc = x1x1_sumc + maxcat;
        x2x1_sumc = x1x1_sumc + maxcat;
        x2x2_sumc = x1x1_sumc + maxcat;
        x2x3_sumc = x1x1_sumc + maxcat;
        x2x4_sumc = x1x1_sumc + maxcat;
        x3x1_sumc = x1x1_sumc + maxcat;
        x3x2_sumc = x1x1_sumc + maxcat;
        x3x3_sumc = x1x1_sumc + maxcat;
        x3x4_sumc = x1x1_sumc + maxcat;
        x4x1_sumc = x1x1_sumc + maxcat;
        x4x2_sumc = x1x1_sumc + maxcat;
        x4x3_sumc = x1x1_sumc + maxcat;
        x4x4_sumc = x1x1_sumc + maxcat;
        x1y_sumc = x4x4_sumc + maxcat;
        x2y_sumc = x1y_sumc + maxcat;
        x3y_sumc = x2y_sumc + maxcat;
        x4y_sumc = x3y_sumc + maxcat;
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
    int k; 
    double x1x1_sum = 0., x1x2_sum = 0., x1x3_sum = 0., x1x4_sum = 0., x2x1_sum = 0., x2x2_sum = 0., x2x3_sum = 0., x2x4_sum = 0., x3x1_sum = 0., x3x2_sum = 0., x3x3_sum = 0., x3x4_sum = 0., x4x1_sum = 0., x4x2_sum = 0., x4x3_sum = 0., x4x4_sum = 0.;
    double x1y_sum = 0., x2y_sum = 0., x3y_sum = 0., x4y_sum = 0.;
    //x: IV, z: y, y: treatment
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

      //finding determinant
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
   
    if (det != 0){
//    det = 1.0 / det;

    for (k = 0; k < 16; k++){
        invOut[k] = inv[k] / det;
    }
    bhat_0 = invOut[0] * x1y_sum + invOut[1] * x2y_sum + invOut[2] * x3y_sum + invOut[3] * x4y_sum;
    bhat_1 = invOut[4] * x1y_sum + invOut[5] * x2y_sum + invOut[6] * x3y_sum + invOut[7] * x4y_sum;
    bhat_2 = invOut[8] * x1y_sum + invOut[9] * x2y_sum + invOut[10] * x3y_sum + invOut[11] * x4y_sum;
    bhat_3 = invOut[12] * x1y_sum + invOut[13] * x2y_sum + invOut[14] * x3y_sum + invOut[15] * x4y_sum;

//    for (i = 0; i < n; i++) {
//        error2 += (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) * (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) / (n - 4); 
//    }
    error2 = (bhat_0*bhat_0 + 2*bhat_0*bhat_1*x1x2_sum + 2*bhat_0*bhat_2*x1x3_sum + 2*bhat_0*bhat_3*x1x4_sum 
              - 2*bhat_0*x1y_sum + bhat_1*bhat_1*x2x2_sum + 2*bhat_1*bhat_2*x2x3_sum + 2*bhat_1*bhat_3*x2x4_sum 
              - 2*bhat_1*x2y_sum + bhat_2*bhat_2*x3x3_sum + 2*bhat_2*bhat_3*x3x4_sum - 2*bhat_2*x3y_sum + bhat_3*bhat_3*x4x4_sum 
              - 2*bhat_3*x4y_sum + yy_sum)/n;
           
    var3 = error2 * invOut[15];   
    } else {
    //x: IV, z: y, y: treatment
    //bhat_3 = (x1y1z_sum/(x1x4_sum) - x1y0z_sum/(x1x3_sum-x1x4_sum)) - (x0y1z_sum/(x1x2_sum-x1x4_sum) - x0y0z_sum/(x1x1_sum-x1x2_sum-x1x3_sum+x1x4_sum));      
    bhat_3 = 0;
    var3 = 1000000;
    Rprintf("CTss Denominator is zero.\n");
    }   
    Rprintf("CTss det, bhat_3 and var3 is %.2f, %.2f, %.2f\n", det, bhat_3, var3 );   
    alpha_1 = (n * xz_sum - x_sum * z_sum) / (n * xy_sum - x_sum * y_sum);
    //effect = alpha_1;
    effect = bhat_3;
    alpha_0 = (z_sum - alpha_1 * y_sum) / n;
    beta_1 = (n * xy_sum - x_sum * y_sum) / (n * xx_sum - x_sum * x_sum);
    beta_0 = (y_sum - beta_1 * x_sum) / n;

    *tr_mean = temp1 / ttreat;
    *con_mean = temp0 / (twt - ttreat);
    *value = effect;
    
    numerator = zz_sum + n * alpha_0 * alpha_0 + alpha_1 * alpha_1 * yy_sum - 2 * alpha_0 * z_sum - 2 * alpha_1 * yz_sum + 2 * alpha_0 * alpha_1 * y_sum;
    denominator = n * beta_0 * beta_0 + beta_1 * beta_1 * xx_sum + y_sum * y_sum / n + 2 * beta_0 * beta_1 * x_sum - 2 * beta_0 * y_sum - 2 * beta_1 * x_sum * y_sum / n;
    //*risk = 4 * twt * max_y * max_y - alpha * twt * effect * effect + (1 - alpha) * (1 + train_to_est_ratio) * twt * (numerator / denominator);
    *risk = 4 * twt * max_y * max_y - alpha * twt * effect * effect + (1 - alpha) * (1 + train_to_est_ratio) * twt * (var3);
// PARAMETER!    
    if(abs(n * xy_sum - x_sum * y_sum) <= 0 * n * n){
        //effect = temp1 / ttreat - temp0 / (twt - ttreat);  
        //*value = effect;
        tr_var = tr_sqr_sum / ttreat - temp1 * temp1 / (ttreat * ttreat);
        con_var = con_sqr_sum / (twt - ttreat) - temp0 * temp0 / ((twt - ttreat) * (twt - ttreat));
        //*risk = 4 * twt * max_y * max_y - alpha * twt * effect * effect + 
        //(1 - alpha) * (1 + train_to_est_ratio) * twt * (tr_var /ttreat  + con_var / (twt - ttreat));
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
  
    int k;
    double right_x1x1_sum = 0., right_x1x2_sum = 0., right_x1x3_sum = 0., right_x1x4_sum = 0., right_x2x1_sum = 0., right_x2x2_sum = 0., right_x2x3_sum = 0., right_x2x4_sum = 0., right_x3x1_sum = 0., right_x3x2_sum = 0., right_x3x3_sum = 0., right_x3x4_sum = 0., right_x4x1_sum = 0., right_x4x2_sum = 0., right_x4x3_sum = 0., right_x4x4_sum = 0.;
    double right_x1y_sum = 0., right_x2y_sum = 0., right_x3y_sum = 0., right_x4y_sum = 0.;
    double left_x1x1_sum = 0., left_x1x2_sum = 0., left_x1x3_sum = 0., left_x1x4_sum = 0., left_x2x1_sum = 0., left_x2x2_sum = 0., left_x2x3_sum = 0., left_x2x4_sum = 0., left_x3x1_sum = 0., left_x3x2_sum = 0., left_x3x3_sum = 0., left_x3x4_sum = 0., left_x4x1_sum = 0., left_x4x2_sum = 0., left_x4x3_sum = 0., left_x4x4_sum = 0.;
    double left_x1y_sum = 0., left_x2y_sum = 0., left_x3y_sum = 0., left_x4y_sum = 0.;
    float m[16], inv[16], invOut[16];
    double det;
    double bhat_0 = 0., bhat_1 = 0., bhat_2 = 0., bhat_3 = 0.;
    double error2 = 0., var3 = 0.;
  
    //double *xz_sum, *xy_sum, *x_sum, *y_sum, *z_sum, *yz_sum, *xx_sum, *yy_sum, *zz_sum; //declare double for categorical
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
//    det = 1.0 / det; //may need to have if(det = 0)

    for (k = 0; k < 16; k++){
        invOut[k] = inv[k] / det;

    }
    bhat_0 = invOut[0] * right_x1y_sum + invOut[1] * right_x2y_sum + invOut[2] * right_x3y_sum + invOut[3] * right_x4y_sum;
    bhat_1 = invOut[4] * right_x1y_sum + invOut[5] * right_x2y_sum + invOut[6] * right_x3y_sum + invOut[7] * right_x4y_sum;
    bhat_2 = invOut[8] * right_x1y_sum + invOut[9] * right_x2y_sum + invOut[10] * right_x3y_sum + invOut[11] * right_x4y_sum;
    bhat_3 = invOut[12] * right_x1y_sum + invOut[13] * right_x2y_sum + invOut[14] * right_x3y_sum + invOut[15] * right_x4y_sum;

//    for (i = 0; i < n; i++) {
//        error2 += (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) * (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) / (n - 4); 
//    }
    error2 = (bhat_0*bhat_0 + 2*bhat_0*bhat_1*right_x1x2_sum + 2*bhat_0*bhat_2*right_x1x3_sum + 2*bhat_0*bhat_3*right_x1x4_sum 
            - 2*bhat_0*right_x1y_sum + bhat_1*bhat_1*right_x2x2_sum + 2*bhat_1*bhat_2*right_x2x3_sum + 2*bhat_1*bhat_3*right_x2x4_sum 
            - 2*bhat_1*right_x2y_sum + bhat_2*bhat_2*right_x3x3_sum + 2*bhat_2*bhat_3*right_x3x4_sum - 2*bhat_2*right_x3y_sum 
            + bhat_3*bhat_3*right_x4x4_sum - 2*bhat_3*right_x4y_sum + right_yy_sum)/right_n;
           
    var3 = error2 * invOut[15]; 

    } else {
    //x: IV, z: y, y: treatment
    //bhat_3 = (right_x1y1z_sum/(right_x1x4_sum) - right_x1y0z_sum/(right_x1x3_sum-right_x1x4_sum)) 
    //        - (right_x0y1z_sum/(right_x1x2_sum-right_x1x4_sum) - right_x0y0z_sum/(right_x1x1_sum-right_x1x2_sum-right_x1x3_sum+right_x1x4_sum)); 
    //Rprintf("Node effect is %.2f", bhat_3);
    bhat_3 = 0;
    var3 = 1000000;
    Rprintf("CT Denominator is zero.\n");
    }   
    Rprintf("CT det, bhat_3 and var3 is %.2f, %.2f, %.2f\n", det, bhat_3, var3 );       
    alpha_1 = (right_n * right_xz_sum - right_x_sum * right_z_sum) / (right_n * right_xy_sum - right_x_sum * right_y_sum);
    alpha_0 = (right_z_sum - alpha_1 * right_y_sum) / right_n;
    beta_1 = (right_n * right_xy_sum - right_x_sum * right_y_sum) / (right_n * right_xx_sum - right_x_sum * right_x_sum);
    beta_0 = (right_y_sum - beta_1 * right_x_sum) / right_n;
    //temp = alpha_1;
    temp = bhat_3;
    numerator = right_zz_sum + right_n * alpha_0 * alpha_0 + alpha_1 * alpha_1 * right_yy_sum - 2 * alpha_0 * right_z_sum - 2 * alpha_1 * right_yz_sum + 2 * alpha_0 * alpha_1 * right_y_sum;
    denominator = right_n * beta_0 * beta_0 + beta_1 * beta_1 * right_xx_sum + right_y_sum * right_y_sum / right_n + 2 * beta_0 * beta_1 * right_x_sum - 2 * beta_0 * right_y_sum - 2 * beta_1 * right_x_sum * right_y_sum / right_n;
    //node_effect = alpha * temp * temp * right_wt - (1 - alpha) * (1 + train_to_est_ratio) 
    //    * right_wt * (numerator / denominator);
    node_effect = alpha * temp * temp * right_wt - (1 - alpha) * (1 + train_to_est_ratio) 
        * right_wt * (var3);
    
// PARAMETER!        
    if(abs(right_n * right_xy_sum - right_x_sum * right_y_sum) <= 0 * right_n * right_n){
            //temp = right_tr_sum / right_tr - (right_sum - right_tr_sum) / (right_wt - right_tr);
            tr_var = right_tr_sqr_sum / right_tr - right_tr_sum * right_tr_sum / (right_tr * right_tr);
            con_var = (right_sqr_sum - right_tr_sqr_sum) / (right_wt - right_tr)
                - (right_sum - right_tr_sum) * (right_sum - right_tr_sum) 
                / ((right_wt - right_tr) * (right_wt - right_tr));
            //node_effect = alpha * temp * temp * right_wt - (1 - alpha) * (1 + train_to_est_ratio) 
            //    * right_wt * (tr_var / right_tr  + con_var / (right_wt - right_tr));
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
                    * (left_tr_var / left_tr + left_con_var / (left_wt - left_tr));
                continue;}
                

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
                            (right_tr_var / right_tr + right_con_var / (right_wt - right_tr));
                continue;}
                

                
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
        
        Rprintf("Entered CT Categorical.\n");
        
        for (i = 0; i < nclass; i++) {
            countn[i] = 0;
            wts[i] = 0;
            trs[i] = 0;
            sums[i] = 0;
            wtsums[i] = 0;
            trsums[i] = 0;
            wtsqrsums[i] = 0;
            trsqrsums[i] = 0;
                
            xz_sumc[i] = 0;
            xy_sumc[i] = 0;
            x_sumc[i] = 0;
            y_sumc[i] = 0;
            z_sumc[i] = 0;
            yz_sumc[i] = 0;
            xx_sumc[i] = 0;
            yy_sumc[i] = 0;
            zz_sumc[i] = 0;    
                         
       //begin of dd
        x1x1_sumc[i] = 0;
        x1x2_sumc[i] = 0;
        x1x3_sumc[i] = 0;  
        x1x4_sumc[i] = 0;
        x2x1_sumc[i] = 0;
        x2x2_sumc[i] = 0;
        x2x3_sumc[i] = 0;  
        x2x4_sumc[i] = 0;
        x3x1_sumc[i] = 0;
        x3x2_sumc[i] = 0;;
        x3x3_sumc[i] = 0;  
        x3x4_sumc[i] = 0; 
        x4x1_sumc[i] = 0;
        x4x2_sumc[i] = 0;
        x4x3_sumc[i] = 0;   
        x4x4_sumc[i] = 0; 
        x1y_sumc[i] = 0;
        x2y_sumc[i] = 0;
        x3y_sumc[i] = 0;  
        x4y_sumc[i] = 0;
        }
        
        /* rank the classes by treatment effect */
        for (i = 0; i < n; i++) {
            j = (int) x[i] - 1;
            Rprintf("i, j is %.2d, %.2d\n", i, j);
            countn[j]++;
            wts[j] += wt[i];
            trs[j] += wt[i] * treatment[i];
            sums[j] += *y[i];
            wtsums[j] += *y[i] * wt[i];
            trsums[j] += *y[i] * wt[i] * treatment[i];
            wtsqrsums[j] += (*y[i]) * (*y[i]) * wt[i];
            trsqrsums[j] +=  (*y[i]) * (*y[i]) * wt[i] * treatment[i];
                
            xz_sumc[j] += *y[i] * IV[i];
            xy_sumc[j] += treatment[i] * IV[i];
            x_sumc[j] += IV[i];
            y_sumc[j] += treatment[i];
            z_sumc[j] += *y[i];
            yz_sumc[j] += *y[i] * treatment[i];
            xx_sumc[j] += IV[i] * IV[i];
            yy_sumc[j] += treatment[i] * treatment[i];
            zz_sumc[j] += *y[i] * *y[i];
                           
            x1x1_sumc[j] += 1 * 1;
            x1x2_sumc[j] += 1 * treatment[i];
            x1x3_sumc[j] += 1 * IV[i];   
            x1x4_sumc[j] += 1 * IV[i] * treatment[i]; 
            x2x1_sumc[j] += treatment[i] * 1;
            x2x2_sumc[j] += treatment[i] * treatment[i];
            x2x3_sumc[j] += treatment[i] * IV[i];   
            x2x4_sumc[j] += treatment[i] * IV[i] * treatment[i]; 
            x3x1_sumc[j] += IV[i] * 1;
            x3x2_sumc[j] += IV[i] * treatment[i];
            x3x3_sumc[j] += IV[i] * IV[i];   
            x3x4_sumc[j] += IV[i] * IV[i] * treatment[i];  
            x4x1_sumc[j] += IV[i] * treatment[i] * 1; 
            x4x2_sumc[j] += IV[i] * treatment[i] * treatment[i];
            x4x3_sumc[j] += IV[i] * treatment[i] * IV[i];   
            x4x4_sumc[j] += IV[i] * treatment[i] * IV[i] * treatment[i];  
            x1y_sumc[j] += *y[i];
            x2y_sumc[j] += *y[i] * treatment[i];
            x3y_sumc[j] += *y[i] * IV[i];  
            x4y_sumc[j] += *y[i] * IV[i] * treatment[i];  
        
        Rprintf("x1-3x1-3 is %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f \n", x1x1_sumc[j], x1x2_sumc[j], x1x3_sumc[j], x1x4_sumc[j], x2x2_sumc[j], x2x3_sumc[j], x2x4_sumc[j], x3x3_sumc[j], x3x4_sumc[j], x4x4_sumc[j]);      
        }
        for (i = 0; i < nclass; i++) {
            if (countn[i] > 0) {
                tsplit[i] = RIGHT;
        
    m[0] = x1x1_sumc[i];
    m[1] = x1x2_sumc[i];
    m[2] = x1x3_sumc[i];
    m[3] = x1x4_sumc[i];
    m[4] = x2x1_sumc[i];
    m[5] = x2x2_sumc[i];
    m[6] = x2x3_sumc[i];
    m[7] = x2x4_sumc[i];
    m[8] = x3x1_sumc[i];
    m[9] = x3x2_sumc[i];
    m[10] = x3x3_sumc[i];
    m[11] = x3x4_sumc[i];
    m[12] = x4x1_sumc[i];
    m[13] = x4x2_sumc[i];
    m[14] = x4x3_sumc[i];    
    m[15] = x4x4_sumc[i];   
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
    Rprintf("m0-15 is %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n", m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);                     
    if (det != 0){
//  det = 1.0 / det;
    for (k = 0; k < 16; k++){
        invOut[k] = inv[k] / det;
    }
    bhat_0 = invOut[0] * x1y_sumc[i] + invOut[1] * x2y_sumc[i] + invOut[2] * x3y_sumc[i] + invOut[3] * x4y_sumc[i];
    bhat_1 = invOut[4] * x1y_sumc[i] + invOut[5] * x2y_sumc[i] + invOut[6] * x3y_sumc[i] + invOut[7] * x4y_sumc[i];
    bhat_2 = invOut[8] * x1y_sumc[i] + invOut[9] * x2y_sumc[i] + invOut[10] * x3y_sumc[i] + invOut[11] * x4y_sumc[i];
    bhat_3 = invOut[12] * x1y_sumc[i] + invOut[13] * x2y_sumc[i] + invOut[14] * x3y_sumc[i] + invOut[15] * x4y_sumc[i];

    error2 = (bhat_0*bhat_0 + 2*bhat_0*bhat_1*x1x2_sumc[i] + 2*bhat_0*bhat_2*x1x3_sumc[i] + 2*bhat_0*bhat_3*x1x4_sumc[i] 
              - 2*bhat_0*x1y_sumc[i] + bhat_1*bhat_1*x2x2_sumc[i] + 2*bhat_1*bhat_2*x2x3_sumc[i] + 2*bhat_1*bhat_3*x2x4_sumc[i] 
              - 2*bhat_1*x2y_sumc[i] + bhat_2*bhat_2*x3x3_sumc[i] + 2*bhat_2*bhat_3*x3x4_sumc[i] - 2*bhat_2*x3y_sumc[i] + bhat_3*bhat_3*x4x4_sumc[i] 
              - 2*bhat_3*x4y_sumc[i] + yy_sumc[i])/n;
           
    var3 = error2 * invOut[15];   
    } else {
    //x: IV, z: y, y: treatment
    bhat_3 = 0;
    var3 = 1000000;
    //Rprintf("Node Denominator is zero.\n");
    }         
    Rprintf("Node det, bhat_3 and var3 is %.2f, %.2f, %.2f\n", det, bhat_3, var3 );          
        //treatment_effect[i] = (countn[j] * xz_sumc[j] - x_sumc[j] * z_sumc[j]) / (countn[j] * xy_sumc[j] - x_sumc[j] * y_sumc[j]); //alpha_1
        //treatment_effect[i] = (countn[i] - x_sumc[i] * z_sumc[i]) / (countn[i] - x_sumc[i] * y_sumc[i]); //is j or i?
          treatment_effect[i] = bhat_3;
        //        treatment_effect[i] = trsums[j] / trs[j] - (wtsums[j] - trsums[j]) / (wts[j] - trs[j]); //this is wrong
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
//updated below codes, because j looks to be class index.             
            left_xz_sum += xz_sumc[j];
            right_xz_sum -= xz_sumc[j];
            left_xy_sum += xy_sumc[j];
            right_xy_sum -= xy_sumc[j];
            left_x_sum += x_sumc[j];
            right_x_sum -= x_sumc[j];
            left_y_sum += y_sumc[j];
            right_y_sum -= y_sumc[j];
            left_z_sum += z_sumc[j];
            right_z_sum -= z_sumc[j];
            left_yz_sum += yz_sumc[j];
            right_yz_sum -= yz_sumc[j];
            left_xx_sum += xx_sumc[j];
            right_xx_sum -= xx_sumc[j];
            left_yy_sum += yy_sumc[j];
            right_yy_sum -= yy_sumc[j];
            left_zz_sum += zz_sumc[j];
            right_zz_sum -= zz_sumc[j];
                
          
            left_x1x1_sum += x1x1_sumc[j];
            right_x1x1_sum -= x1x1_sumc[j];
            left_x1x2_sum += x1x2_sumc[j];
            right_x1x2_sum -= x1x2_sumc[j];
            left_x1x3_sum += x1x3_sumc[j]; 
            right_x1x3_sum -= x1x3_sumc[j];  
            left_x1x4_sum += x1x4_sumc[j];  
            right_x1x4_sum -= x1x4_sumc[j];
            left_x2x1_sum += x2x1_sumc[j];
            right_x2x1_sum -= x2x1_sumc[j];
            left_x2x2_sum += x2x2_sumc[j];
            right_x2x2_sum -= x2x2_sumc[j];
            left_x2x3_sum += x2x3_sumc[j];  
            right_x2x3_sum -= x2x3_sumc[j];
            left_x2x4_sum += x2x4_sumc[j];
            right_x2x4_sum -= x2x4_sumc[j];
            left_x3x1_sum += x3x1_sumc[j];
            right_x3x1_sum -= x3x1_sumc[j];
            left_x3x2_sum += x3x2_sumc[j];
            right_x3x2_sum -= x3x2_sumc[j];
            left_x3x3_sum += x3x3_sumc[j];
            right_x3x3_sum -= x3x3_sumc[j];
            left_x3x4_sum += x3x4_sumc[j];
            right_x3x4_sum -= x3x4_sumc[j]; 
            left_x4x1_sum += x4x1_sumc[j];
            right_x4x1_sum -= x4x1_sumc[j];
            left_x4x2_sum += x4x2_sumc[j];
            right_x4x2_sum -= x4x2_sumc[j];
            left_x4x3_sum += x4x3_sumc[j];
            right_x4x3_sum -= x4x3_sumc[j];
            left_x4x4_sum += x4x4_sumc[j]; 
            right_x4x4_sum -= x4x4_sumc[j]; 
            left_x1y_sum += x1y_sumc[j];
            right_x1y_sum -= x1y_sumc[j];
            left_x2y_sum += x2y_sumc[j];
            right_x2y_sum -= x2y_sumc[j];
            left_x3y_sum += x3y_sumc[j];
            right_x3y_sum -= x3y_sumc[j];
            left_x4y_sum += x4y_sumc[j];
            right_x4y_sum -= x4y_sumc[j];
          
            if (left_n >= edge && right_n >= edge &&
                (int) left_tr >= min_node_size &&
                (int) left_wt - (int) left_tr >= min_node_size &&
                (int) right_tr >= min_node_size &&
                (int) right_wt - (int) right_tr >= min_node_size) {
                
              
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
    for (k = 0; k < 16; k++){
        invOut[k] = inv[k] / det;
    }
    bhat_0 = invOut[0] * left_x1y_sum + invOut[1] * left_x2y_sum + invOut[2] * left_x3y_sum + invOut[3] * left_x4y_sum;
    bhat_1 = invOut[4] * left_x1y_sum + invOut[5] * left_x2y_sum + invOut[6] * left_x3y_sum + invOut[7] * left_x4y_sum;
    bhat_2 = invOut[8] * left_x1y_sum + invOut[9] * left_x2y_sum + invOut[10] * left_x3y_sum + invOut[11] * left_x4y_sum;
    bhat_3 = invOut[12] * left_x1y_sum + invOut[13] * left_x2y_sum + invOut[14] * left_x3y_sum + invOut[15] * left_x4y_sum;
    error2 = (bhat_0*bhat_0 + 2*bhat_0*bhat_1*left_x1x2_sum + 2*bhat_0*bhat_2*left_x1x3_sum + 2*bhat_0*bhat_3*left_x1x4_sum 
              - 2*bhat_0*left_x1y_sum + bhat_1*bhat_1*left_x2x2_sum + 2*bhat_1*bhat_2*left_x2x3_sum + 2*bhat_1*bhat_3*left_x2x4_sum 
              - 2*bhat_1*left_x2y_sum + bhat_2*bhat_2*left_x3x3_sum + 2*bhat_2*bhat_3*left_x3x4_sum - 2*bhat_2*left_x3y_sum 
              + bhat_3*bhat_3*left_x4x4_sum - 2*bhat_3*left_x4y_sum + left_yy_sum)/left_n;
    var3 = error2 * invOut[15];   
    } else {
    bhat_3 = 0;
    var3 = 1000000;
    //Rprintf("Left Denominator is zero.\n");
    }                    
           
    Rprintf("Left node det, bhat_3 and var3 is %.2f, %.2f, %.2f\n", det, bhat_3, var3 );        
              
                alpha_1 = (left_n * left_xz_sum - left_x_sum * left_z_sum) / (left_n * left_xy_sum - left_x_sum * left_y_sum);
                alpha_0 = (left_z_sum - alpha_1 * left_y_sum) / left_n;
                beta_1 = (left_n * left_xy_sum - left_x_sum * left_y_sum) / (left_n * left_xx_sum - left_x_sum * left_x_sum);
                beta_0 = (left_y_sum - beta_1 * left_x_sum) / left_n;
                //left_temp = alpha_1;
                left_temp = bhat_3; 
                numerator = left_zz_sum + left_n * alpha_0 * alpha_0 + alpha_1 * alpha_1 * left_yy_sum - 2 * alpha_0 * left_z_sum - 2 * alpha_1 * left_yz_sum + 2 * alpha_0 * alpha_1 * left_y_sum;
                denominator = left_n * beta_0 * beta_0 + beta_1 * beta_1 * left_xx_sum + left_y_sum * left_y_sum / left_n + 2 * beta_0 * beta_1 * left_x_sum - 2 * beta_0 * left_y_sum - 2 * beta_1 * left_x_sum * left_y_sum / left_n;
                //left_effect = alpha * left_temp * left_temp * left_wt - (1 - alpha) * (1 + train_to_est_ratio) 
                //    * left_wt * (numerator / denominator);
                left_effect = alpha * left_temp * left_temp * left_wt - (1 - alpha) * (1 + train_to_est_ratio) 
                    * left_wt * (var3);    
                //left_temp = left_tr_sum / left_tr - (left_sum - left_tr_sum) 
                //    / (left_wt - left_tr);
                
                //left_tr_var = left_tr_sqr_sum / left_tr 
                //    - left_tr_sum  * left_tr_sum / (left_tr * left_tr);
                //left_con_var = (left_sqr_sum - left_tr_sqr_sum) / (left_wt - left_tr)  
                //   - (left_sum - left_tr_sum) * (left_sum - left_tr_sum)
                //   / ((left_wt - left_tr) * (left_wt - left_tr));       
                //left_effect = alpha * left_temp * left_temp * left_wt
                //    - (1 - alpha) * (1 + train_to_est_ratio) * left_wt * 
                //        (left_tr_var / left_tr + left_con_var / (left_wt - left_tr));
               
              
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
    for (k = 0; k < 16; k++){
        invOut[k] = inv[k] / det;
    }
    bhat_0 = invOut[0] * right_x1y_sum + invOut[1] * right_x2y_sum + invOut[2] * right_x3y_sum + invOut[3] * right_x4y_sum;
    bhat_1 = invOut[4] * right_x1y_sum + invOut[5] * right_x2y_sum + invOut[6] * right_x3y_sum + invOut[7] * right_x4y_sum;
    bhat_2 = invOut[8] * right_x1y_sum + invOut[9] * right_x2y_sum + invOut[10] * right_x3y_sum + invOut[11] * right_x4y_sum;
    bhat_3 = invOut[12] * right_x1y_sum + invOut[13] * right_x2y_sum + invOut[14] * right_x3y_sum + invOut[15] * right_x4y_sum;
    error2 = (bhat_0*bhat_0 + 2*bhat_0*bhat_1*right_x1x2_sum + 2*bhat_0*bhat_2*right_x1x3_sum + 2*bhat_0*bhat_3*right_x1x4_sum 
              - 2*bhat_0*right_x1y_sum + bhat_1*bhat_1*right_x2x2_sum + 2*bhat_1*bhat_2*right_x2x3_sum 
              + 2*bhat_1*bhat_3*right_x2x4_sum - 2*bhat_1*right_x2y_sum + bhat_2*bhat_2*right_x3x3_sum 
              + 2*bhat_2*bhat_3*right_x3x4_sum - 2*bhat_2*right_x3y_sum + bhat_3*bhat_3*right_x4x4_sum 
              - 2*bhat_3*right_x4y_sum + right_yy_sum)/right_n;
    var3 = error2 * invOut[15];   
    } else { 
    //Rprintf("Right Denominator is zero.\n");
    bhat_3 = 0;
    var3 = 1000000;
    }                
    Rprintf("Right node det, bhat_3 and var3 is %.2f, %.2f, %.2f\n", det, bhat_3, var3 );              
                alpha_1 = (right_n * right_xz_sum - right_x_sum * right_z_sum) / (right_n * right_xy_sum - right_x_sum * right_y_sum);
                alpha_0 = (right_z_sum - alpha_1 * right_y_sum) / right_n;
                beta_1 = (right_n * right_xy_sum - right_x_sum * right_y_sum) / (right_n * right_xx_sum - right_x_sum * right_x_sum);
                beta_0 = (right_y_sum - beta_1 * right_x_sum) / right_n;
                //right_temp = alpha_1;
                right_temp = bhat_3; 
                numerator = right_zz_sum + right_n * alpha_0 * alpha_0 + alpha_1 * alpha_1 * right_yy_sum - 2 * alpha_0 * right_z_sum - 2 * alpha_1 * right_yz_sum + 2 * alpha_0 * alpha_1 * right_y_sum;
                denominator = right_n * beta_0 * beta_0 + beta_1 * beta_1 * right_xx_sum + right_y_sum * right_y_sum / right_n + 2 * beta_0 * beta_1 * right_x_sum - 2 * beta_0 * right_y_sum - 2 * beta_1 * right_x_sum * right_y_sum / right_n;
                //right_effect = alpha * right_temp * right_temp * right_wt - (1 - alpha) * (1 + train_to_est_ratio) 
                //    * right_wt * (numerator / denominator);
                right_effect = alpha * right_temp * right_temp * right_wt - (1 - alpha) * (1 + train_to_est_ratio) 
                    * right_wt * (var3);    
                //right_temp = right_tr_sum / right_tr - (right_sum - right_tr_sum) 
                //    / (right_wt - right_tr);
                //right_tr_var = right_tr_sqr_sum / right_tr 
                //    - right_tr_sum * right_tr_sum / (right_tr * right_tr);
                //right_con_var = (right_sqr_sum - right_tr_sqr_sum) / (right_wt - right_tr)
                //    - (right_sum - right_tr_sum) * (right_sum - right_tr_sum) 
                //    / ((right_wt - right_tr) * (right_wt - right_tr));
                //right_effect = alpha * right_temp * right_temp * right_wt
                //        - (1 - alpha) * (1 + train_to_est_ratio) * right_wt *
                //            (right_tr_var / right_tr + right_con_var / (right_wt - right_tr));
                temp = left_effect + right_effect - node_effect;
    Rprintf("Temp and best are %.2f, %.2f\n", temp, best );  
                
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
