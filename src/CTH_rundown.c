/*
 * This rundown function for CTH.
 *
 */
#include "causalTree.h"
#include "node.h"
#include "causalTreeproto.h"

#ifdef NAN
/* NAN is supported */
#endif

void
CTH_rundown(pNode tree, int obs, double *cp, double *xpred, double *xtemp, int k, double alpha, 
            double xtrain_to_est_ratio, double propensity)
{
    int i, obs2 = (obs < 0) ? -(1 + obs) : obs;
    int my_leaf_id;
    pNode otree =  tree;
    pNode otree_tmp = tree;
    pNode tree_tmp = tree;
    
    int opnumber = 0;
    int j, s;
    int tmp_obs, tmp_id;
    double tr_mean, con_mean;
    double tr_sqr_sum, con_sqr_sum;
    double consums, trsums, cons, trs;
    double tr_var, con_var;
    double xz_sum, xy_sum, x_sum, y_sum, z_sum;
    double yz_sum, xx_sum, yy_sum, zz_sum;
    int n;
    double x1x1_sum = 0., x1x2_sum = 0., x1x3_sum = 0., x1x4_sum = 0., x2x1_sum = 0., x2x2_sum = 0., x2x3_sum = 0., x2x4_sum = 0., x3x1_sum = 0., x3x2_sum = 0., x3x3_sum = 0., x3x4_sum = 0., x4x1_sum = 0., x4x2_sum = 0., x4x3_sum = 0., x4x4_sum = 0.;
    double x1y_sum = 0., x2y_sum = 0., x3y_sum = 0., x4y_sum = 0.;
    float m[16], inv[16], invOut[16];
    double det;
    double bhat_0 = 0., bhat_1 = 0., bhat_2 = 0., bhat_3 = 0.;
    double error2 = 0., var3 = 0.;

    /*
     * Now, repeat the following: for the cp of interest, run down the tree
     *   until I find a node with smaller complexity.  The parent node will
     *   not have collapsed, but this split will have, so this is my
     *   predictor.
     */
    for (i = 0; i < ct.num_unique_cp; i++) {
        cons = 0.;
        trs = 0.;
        consums = 0.;
        trsums = 0.;
        tr_sqr_sum = 0.;
        con_sqr_sum = 0.;
	n = 0;
	xz_sum = 0.;
	xy_sum = 0.;
	x_sum = 0.;
	y_sum = 0.;
	z_sum = 0.;
        yz_sum = 0.;
	xx_sum = 0.;
	yy_sum = 0.;
	zz_sum = 0.;    
	x1x1_sum = 0.;
	x1x2_sum = 0.;
	x1x3_sum = 0.;
	x1x4_sum = 0.;
	x2x1_sum = 0.;
	x2x2_sum = 0.;
	x2x3_sum = 0.;
	x2x4_sum = 0.;
	x3x1_sum = 0.;
	x3x2_sum = 0.;
	x3x3_sum = 0.;
	x3x4_sum = 0.;
	x4x1_sum = 0.;
	x4x2_sum = 0.;
	x4x3_sum = 0.;
	x4x4_sum = 0.;
        x1y_sum = 0.;
	x2y_sum = 0.;
	x3y_sum = 0.;
	x4y_sum = 0.;

        while (cp[i] < tree->complexity) {
	        tree = branch(tree, obs);
	        if (tree == 0)
		        goto oops;
	        otree = tree;
	    }
	    xpred[i] = tree->response_est[0];
        my_leaf_id = tree->id;
        
        for (s = k; s < ct.n; s++) {
            tree_tmp = otree_tmp;
            j = ct.sorts[0][s];
            tmp_obs = (j < 0) ? -(1 + j) : j;
            while (cp[i] < tree_tmp->complexity) {
                tree_tmp = branch(tree_tmp, tmp_obs);
            }
            tmp_id = tree_tmp->id;
            if (tmp_id == my_leaf_id) {
                if (ct.treatment[tmp_obs] == 0) {
                    cons += ct.wt[tmp_obs];
                    consums += *ct.ydata[tmp_obs] * ct.wt[tmp_obs];
                    con_sqr_sum += (*ct.ydata[tmp_obs]) * (*ct.ydata[tmp_obs]) * ct.wt[tmp_obs];
                } else {
                    trs += ct.wt[tmp_obs];
                    trsums += *ct.ydata[tmp_obs] * ct.wt[tmp_obs];
                    tr_sqr_sum += (*ct.ydata[tmp_obs]) * (*ct.ydata[tmp_obs]) * ct.wt[tmp_obs];
                }
		n++;
		xz_sum += ct.IV[tmp_obs] * *ct.ydata[tmp_obs];
                xy_sum += ct.IV[tmp_obs] * ct.treatment[tmp_obs];
		x_sum += ct.IV[tmp_obs];
                y_sum += ct.treatment[tmp_obs];
                z_sum += *ct.ydata[tmp_obs];
                yz_sum += *ct.ydata[tmp_obs] * ct.treatment[tmp_obs];
                xx_sum += ct.IV[tmp_obs] * ct.IV[tmp_obs];
                yy_sum += ct.treatment[tmp_obs] * ct.treatment[tmp_obs];
                zz_sum += *ct.ydata[tmp_obs] * *ct.ydata[tmp_obs];
	x1x1_sum += 1 * 1;
        x1x2_sum += 1 * ct.treatment[tmp_obs];
        x1x3_sum += 1 * ct.IV[tmp_obs];   
        x1x4_sum += 1 * ct.IV[tmp_obs] * ct.treatment[tmp_obs]; 
        x2x1_sum += ct.treatment[tmp_obs] * 1;
        x2x2_sum += ct.treatment[tmp_obs] * ct.treatment[tmp_obs];
        x2x3_sum += ct.treatment[tmp_obs] * ct.IV[tmp_obs];   
        x2x4_sum += ct.treatment[tmp_obs] * ct.IV[tmp_obs] * ct.treatment[tmp_obs]; 
        x3x1_sum += ct.IV[tmp_obs] * 1;
        x3x2_sum += ct.IV[tmp_obs] * ct.treatment[tmp_obs];
        x3x3_sum += ct.IV[tmp_obs] * ct.IV[tmp_obs];   
        x3x4_sum += ct.IV[tmp_obs] * ct.IV[tmp_obs] * ct.treatment[tmp_obs];  
        x4x1_sum += ct.IV[tmp_obs] * ct.treatment[tmp_obs] * 1; 
        x4x2_sum += ct.IV[tmp_obs] * ct.treatment[tmp_obs] * ct.treatment[tmp_obs];
        x4x3_sum += ct.IV[tmp_obs] * ct.treatment[tmp_obs] * ct.IV[tmp_obs];   
        x4x4_sum += ct.IV[tmp_obs] * ct.treatment[tmp_obs] * ct.IV[tmp_obs] * ct.treatment[tmp_obs];  
        x1y_sum += *ct.ydata[tmp_obs];
        x2y_sum += *ct.ydata[tmp_obs] * ct.treatment[tmp_obs];
        x3y_sum += *ct.ydata[tmp_obs] * ct.IV[tmp_obs];  
        x4y_sum += *ct.ydata[tmp_obs] * ct.IV[tmp_obs] * ct.treatment[tmp_obs];  
            }
        }

        if (trs == 0) {
            tr_mean = tree->parent->xtreatMean[0];
            tr_var = 0;
        } else {
            tr_mean = trsums / trs;
            tree->xtreatMean[0] = tr_mean;
            tr_var = tr_sqr_sum / trs - tr_mean * tr_mean;
        }
        
        if (cons == 0) {
            con_mean = tree->parent->xcontrolMean[0];
            con_var = 0;
        } else {
            con_mean = consums / cons;
            tree->xcontrolMean[0] = con_mean;
            con_var = con_sqr_sum / cons - con_mean * con_mean;
        }
        
        //xtemp[i] = (*ct_xeval)(ct.ydata[obs2], ct.wt[obs2], ct.treatment[obs2], tr_mean, 
        //            con_mean, trs, cons, alpha, xtrain_to_est_ratio, propensity);
//	double alpha_1;
// PARAMETER!	    
//	if (abs(n * xy_sum - x_sum * y_sum) <= 0 * n * n){
//		alpha_1 = 0.;
//	}
//	else{
//		alpha_1 = (n * xz_sum - x_sum * z_sum) / (n * xy_sum - x_sum * y_sum);
//	}
//        double effect = alpha_1;
//        double alpha_0 = (z_sum - alpha_1 * y_sum) / n;
//	  double beta_1;
//	if (n * xx_sum - x_sum * x_sum == 0){
//		beta_1 = 0.;
//	}
//	else{
//		beta_1 = (n * xy_sum - x_sum * y_sum) / (n * xx_sum - x_sum * x_sum);
//	}    
//        double beta_0 = (y_sum - beta_1 * x_sum) / n;

//	double numerator = (ct.ydata[obs2][0] - alpha_0 - alpha_1 * ct.treatment[obs2]) * (ct.ydata[obs2][0] - alpha_0 - alpha_1 * ct.treatment[obs2]);
        //double numerator = zz_sum + n * alpha_0 * alpha_0 + alpha_1 * alpha_1 * yy_sum - 2 * alpha_0 * z_sum - 2 * alpha_1 * yz_sum + 2 * alpha_0 * alpha_1 * y_sum;
//        double denominator = n * beta_0 * beta_0 + beta_1 * beta_1 * xx_sum + y_sum * y_sum / n + 2 * beta_0 * beta_1 * x_sum - 2 * beta_0 * y_sum - 2 * beta_1 * x_sum * y_sum / n;
//	double tmp;
//	if (n > 2 && denominator!=0) {
//            tmp = numerator / denominator / (n - 2);
//        } else {
//            tmp = 0.;
//        }  
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
    det = 1.0 / det;

    for (i = 0; i < 16; i++){
        invOut[i] = inv[i] * det;
    }
    bhat_0 = invOut[0] * x1y_sum + invOut[1] * x2y_sum + invOut[2] * x3y_sum + invOut[3] * x4y_sum;
    bhat_1 = invOut[4] * x1y_sum + invOut[5] * x2y_sum + invOut[6] * x3y_sum + invOut[7] * x4y_sum;
    bhat_2 = invOut[8] * x1y_sum + invOut[9] * x2y_sum + invOut[10] * x3y_sum + invOut[11] * x4y_sum;
    bhat_3 = invOut[12] * x1y_sum + invOut[13] * x2y_sum + invOut[14] * x3y_sum + invOut[15] * x4y_sum;
    error2 = (ct.ydata[obs2][0] - bhat_0 - bhat_1 * ct.treatment[obs2] - bhat_2 * ct.IV[obs2] - bhat_3 * ct.IV[obs2] * ct.treatment[obs2]) * (ct.ydata[obs2][0] - bhat_0 - bhat_1 * ct.treatment[obs2] - bhat_2 * ct.IV[obs2] - bhat_3 * ct.IV[obs2] * ct.treatment[obs2]); 
    //for (i = 0; i < n; i++) {
    //    error2 += (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) * (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) / (n - 4); 
    //}
    var3 = error2 * invOut[15];   
    } else{
    bhat_3 = 0;
    var3 = 0;
    }	
        xtemp[i] = 4 * ct.max_y * ct.max_y - alpha * bhat_3 * bhat_3 + (1 + xtrain_to_est_ratio / (ct.NumXval - 1)) * (1 - alpha) * var3;
    //    xtemp[i] = 4 * ct.max_y * ct.max_y - alpha * effect * effect + (1 + xtrain_to_est_ratio / (ct.NumXval - 1)) * (1 - alpha) * tmp;
    }
    return;

oops:;
    if (ct.usesurrogate < 2) {  /* must have hit a missing value */
	for (i = 0; i < ct.num_unique_cp; i++)
	    xpred[i] = otree->response_est[0];

	xtemp[i] = (*ct_xeval)(ct.ydata[obs2], ct.wt[obs2], ct.treatment[obs2], tr_mean, con_mean);
	Rprintf("oops number %d.\n", opnumber++);
  return;
    }
    warning("Warning message--see rundown.c");
}
