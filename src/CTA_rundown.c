/*
 * This rundown functions You may change it to make it compatibel with other splitting funcitons.
 *
 */
#include "causalTree.h"
#include "node.h"
#include "causalTreeproto.h"

#ifdef NAN
/* NAN is supported */
#endif

void
CTA_rundown(pNode tree, int obs, double *cp, double *xpred, double *xtemp, int k, double alpha)
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
    double consums, trsums, cons, trs;
    double sum_ivy = 0., sum_iv = 0., sum_y = 0.;
    double sum_ivt = 0., sum_t = 0.;
    double x1x1_sum = 0., x1x2_sum = 0., x1x3_sum = 0., x1x4_sum = 0., x2x1_sum = 0., x2x2_sum = 0., x2x3_sum = 0., x2x4_sum = 0., x3x1_sum = 0., x3x2_sum = 0., x3x3_sum = 0., x3x4_sum = 0., x4x1_sum = 0., x4x2_sum = 0., x4x3_sum = 0., x4x4_sum = 0.;
    double x1y_sum = 0., x2y_sum = 0., x3y_sum = 0., x4y_sum = 0.;
    float m[16], inv[16], invOut[16];
    double det;
    double bhat_0 = 0., bhat_1 = 0., bhat_2 = 0., bhat_3 = 0.;
    double error2 = 0.;

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
        
        while (cp[i] < tree->complexity) {
	        tree = branch(tree, obs);
	        if (tree == 0)
		        goto oops;
	        otree = tree;
	    }
	    xpred[i] = tree->response_est[0];
        // now find other samples in the same leaf;
        my_leaf_id = tree->id;
        
        
        for (s = k; s < ct.n; s++) {
            tree_tmp = otree_tmp;
            j = ct.sorts[0][s];
            // test: 
           
            tmp_obs = (j < 0) ? -(1 + j) : j;
            while (cp[i] < tree_tmp->complexity) {
                tree_tmp = branch(tree_tmp, tmp_obs);
            }
            tmp_id = tree_tmp->id;

            if (tmp_id == my_leaf_id) {
                if (ct.treatment[j] == 0) {
                    cons += ct.wt[j];
                    consums += *ct.ydata[j] * ct.wt[j];
		    
                } else {
                    trs += ct.wt[j];
                    trsums += *ct.ydata[j] * ct.wt[j];
                }
		sum_ivy += ct.IV[j] * *ct.ydata[j];
	        sum_iv += ct.IV[j];
	        sum_y += *ct.ydata[j];
		sum_ivt += ct.IV[j] * ct.treatment[j];
		sum_t += ct.treatment[j];
	x1x1_sum += 1 * 1;
        x1x2_sum += 1 * ct.treatment[j];
        x1x3_sum += 1 * ct.IV[j];   
        x1x4_sum += 1 * ct.IV[j] * ct.treatment[j]; 
        x2x1_sum += ct.treatment[j] * 1;
        x2x2_sum += ct.treatment[j] * ct.treatment[j];
        x2x3_sum += ct.treatment[j] * ct.IV[j];   
        x2x4_sum += ct.treatment[j] * ct.IV[j] * ct.treatment[j]; 
	x3x1_sum += ct.IV[j] * 1;
        x3x2_sum += ct.IV[j] * ct.treatment[j];
        x3x3_sum += ct.IV[j] * ct.IV[j];   
        x3x4_sum += ct.IV[j] * ct.IV[j] * ct.treatment[j];  
        x4x1_sum += ct.IV[j] * ct.treatment[j] * 1; 
        x4x2_sum += ct.IV[j] * ct.treatment[j] * ct.treatment[j];
        x4x3_sum += ct.IV[j] * ct.treatment[j] * ct.IV[j];   
        x4x4_sum += ct.IV[j] * ct.treatment[j] * ct.IV[j] * ct.treatment[j];  
        x1y_sum += *ct.ydata[j];
        x2y_sum += *ct.ydata[j] * ct.treatment[j];
        x3y_sum += *ct.ydata[j] * ct.IV[j];  
        x4y_sum += *ct.ydata[j] * ct.IV[j] * ct.treatment[j];  
            }
        }
        
        //calculate tr_mean and con_mean
        if (trs == 0) {
            // want to trace back to tree->parent for tr_mean;
            tr_mean = tree->parent->xtreatMean[0];
        } else {
            tr_mean = trsums / trs;
            tree->xtreatMean[0] = tr_mean;
        }
        
        if (cons == 0) {
            con_mean = tree->parent->xcontrolMean[0];
        } else {
            con_mean = consums / cons;
            tree->xcontrolMean[0] = con_mean;
        }
        
        double tree_tr_mean = tree->treatMean[0];
        double tree_con_mean = tree->controlMean[0];

        //xtemp[i] = (*ct_xeval)(ct.ydata[obs2], ct.wt[obs2], ct.treatment[obs2], 
        //            tr_mean, con_mean, tree_tr_mean, tree_con_mean, alpha);
	//double effect_te = ((cons + trs) * sum_ivy - sum_iv * sum_y) / ((cons + trs) * sum_ivt - sum_iv * sum_t);
	//double effect_tr = tree->response_est[0];
	  double effect_tr = 0.;
	    
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
    det = 1.0 / det;

    for (i = 0; i < 16; i++){
        invOut[i] = inv[i] * det;
    }
    bhat_0 = invOut[0] * x1y_sum + invOut[1] * x2y_sum + invOut[2] * x3y_sum + invOut[3] * x4y_sum;
    bhat_1 = invOut[4] * x1y_sum + invOut[5] * x2y_sum + invOut[6] * x3y_sum + invOut[7] * x4y_sum;
    bhat_2 = invOut[8] * x1y_sum + invOut[9] * x2y_sum + invOut[10] * x3y_sum + invOut[11] * x4y_sum;
    bhat_3 = invOut[12] * x1y_sum + invOut[13] * x2y_sum + invOut[14] * x3y_sum + invOut[15] * x4y_sum;
   //xtemp[i] = 2 * ct.max_y * ct.max_y + effect_tr * effect_tr  -  2 *  effect_tr * effect_te;
    } else {
    bhat_3 = 0;
    }	    
     xtemp[i] = 2 * ct.max_y * ct.max_y + effect_tr * effect_tr  -  2 *  effect_tr * bhat_3;
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
