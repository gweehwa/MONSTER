/* 
 * Do honest causalTree estimation with parameters
 * 
 */
 
#include "causalTree.h"
#include "causalTreeproto.h"

    static void
honest_estimate_causalTree0(const int *dimx, int nnode, int nsplit, const int *dimc, 
                            const int *nnum, const int *nodes2, const int *vnum,
                            const double *split2, const int *csplit2, const int *usesur,
                            int *n1, double *wt1, double *dev1, double *yval1, const double *xdata2, 
                            const double *wt2, const double *treatment2, double *IV2, const double *y2,
                            const int *xmiss2, int *where)
{
    int i, j;
    int n;
    int ncat;
    int node, nspl, var, dir;
    int lcount, rcount;
    int npos;
    double temp;
    const int *nodes[3];
    const double *split[4];
    const int **csplit = NULL, **xmiss;
    const double **xdata;
    double *trs = NULL;
    double *cons = NULL; 
    double *trsums = NULL; 
    double *consums = NULL;
    double *trsqrsums = NULL;
    double *consqrsums = NULL;
    double *x_sum = NULL;
    double *y_sum = NULL;
    double *z_sum = NULL;
    double *xy_sum = NULL;
    double *xz_sum = NULL;
    double *yz_sum = NULL;
    double *xx_sum = NULL;
    double *yy_sum = NULL;
    double *zz_sum = NULL;
    double *x1x1_sum = NULL;
    double *x1x2_sum = NULL;
    double *x1x3_sum = NULL;  
    double *x1x4_sum = NULL;
    double *x2x1_sum = NULL;
    double *x2x2_sum = NULL;
    double *x2x3_sum = NULL;  
    double *x2x4_sum = NULL;
    double *x3x1_sum = NULL;
    double *x3x2_sum = NULL;
    double *x3x3_sum = NULL;
    double *x3x4_sum = NULL;
    double *x4x1_sum = NULL;
    double *x4x2_sum = NULL;
    double *x4x3_sum = NULL;
    double *x4x4_sum = NULL; 
    double *x1y_sum = NULL;
    double *x2y_sum = NULL;
    double *x3y_sum = NULL;
    double *x4y_sum = NULL;
    float *m[16] = NULL, *inv[16] = NULL, *invOut[16] = NULL;
    double *det = NULL;
    double *bhat_0 = NULL, *bhat_1 = NULL, *bhat_2 = NULL, *bhat_3 = NULL;
    double *error2 = NULL, *var3 = NULL;
    int nnodemax = -1;
    int *invertdx = NULL;
    
    trs = (double *) ALLOC(nnode, sizeof(double));
    cons = (double *) ALLOC(nnode, sizeof(double));
    trsums = (double *) ALLOC(nnode, sizeof(double));
    consums = (double *) ALLOC(nnode, sizeof(double));
    trsqrsums = (double *) ALLOC(nnode, sizeof(double));
    consqrsums = (double *) ALLOC(nnode, sizeof(double));
    x_sum = (double *) ALLOC(nnode, sizeof(double));
    y_sum = (double *) ALLOC(nnode, sizeof(double));
    z_sum = (double *) ALLOC(nnode, sizeof(double));
    xy_sum = (double *) ALLOC(nnode, sizeof(double));
    xz_sum = (double *) ALLOC(nnode, sizeof(double));
    yz_sum = (double *) ALLOC(nnode, sizeof(double));
    xx_sum = (double *) ALLOC(nnode, sizeof(double));
    yy_sum = (double *) ALLOC(nnode, sizeof(double));
    zz_sum = (double *) ALLOC(nnode, sizeof(double));
    x1x1_sum = (double *) ALLOC(nnode, sizeof(double));
    x1x2_sum = (double *) ALLOC(nnode, sizeof(double));
    x1x3_sum = (double *) ALLOC(nnode, sizeof(double));
    x1x4_sum = (double *) ALLOC(nnode, sizeof(double));
    x2x1_sum = (double *) ALLOC(nnode, sizeof(double));
    x2x2_sum = (double *) ALLOC(nnode, sizeof(double));
    x2x3_sum = (double *) ALLOC(nnode, sizeof(double)); 
    x2x4_sum = (double *) ALLOC(nnode, sizeof(double));
    x3x1_sum = (double *) ALLOC(nnode, sizeof(double));
    x3x2_sum = (double *) ALLOC(nnode, sizeof(double));
    x3x3_sum = (double *) ALLOC(nnode, sizeof(double));
    x3x4_sum = (double *) ALLOC(nnode, sizeof(double));
    x4x1_sum = (double *) ALLOC(nnode, sizeof(double));
    x4x2_sum = (double *) ALLOC(nnode, sizeof(double));
    x4x3_sum = (double *) ALLOC(nnode, sizeof(double));
    x4x4_sum = (double *) ALLOC(nnode, sizeof(double));
    x1y_sum = (double *) ALLOC(nnode, sizeof(double));
    x2y_sum = (double *) ALLOC(nnode, sizeof(double));
    x3y_sum = (double *) ALLOC(nnode, sizeof(double));
    x4y_sum = (double *) ALLOC(nnode, sizeof(double));

    
    // initialize:
    for (i = 0; i < nnode; i++) {
        trs[i] = 0.;
        cons[i] = 0.;
        trsums[i] = 0.;
        consums[i] = 0.;
        trsqrsums[i] = 0.;
        consqrsums[i] = 0.;
        x_sum[i] = 0.;
        y_sum[i] = 0.;
        z_sum[i] = 0.;
        xy_sum[i] = 0.;
        xz_sum[i] = 0.;
        yz_sum[i] = 0.;
        xx_sum[i] = 0.;
        yy_sum[i] = 0.;
        zz_sum[i] = 0.;
        x1x1_sum[i] = 0.;
        x1x2_sum[i] = 0.;
        x1x3_sum[i] = 0.;  
        x1x4_sum[i] = 0.; 
        x2x1_sum[i] = 0.;
        x2x2_sum[i] = 0.;
        x2x3_sum[i] = 0.;  
        x2x4_sum[i] = 0.;
        x3x1_sum[i] = 0.;
        x3x2_sum[i] = 0.;
        x3x3_sum[i] = 0.; 
        x3x4_sum[i] = 0.; 
        x4x1_sum[i] = 0.;
        x4x2_sum[i] = 0.;
        x4x3_sum[i] = 0.; 
        x4x4_sum[i] = 0.; 
        x1y_sum[i] = 0.;
        x2y_sum[i] = 0.;
        x3y_sum[i] = 0.; 
        x4y_sum[i] = 0.; 
        n1[i] = 0;
        wt1[i] = 0.;
        if (nnum[i] > nnodemax) {
            nnodemax = nnum[i]; 
        }
    }
    
    invertdx = (int *) ALLOC(nnodemax + 1, sizeof(int));
    // construct an invert index:
    for (i = 0; i <= nnodemax + 1; i++) {
        invertdx[i] = -1;
    }
    for (i = 0; i != nnode; i++) {
        invertdx[nnum[i]] = i;
    }
    
    n = dimx[0]; // n = # of obs
    for (i = 0; i < 3; i++) {
        nodes[i] = &(nodes2[nnode * i]);
    }
    
    for(i = 0; i < 4; i++) {
        split[i] = &(split2[nsplit * i]);
    }

    if (dimc[1] > 0) {
        csplit = (const int **) ALLOC((int) dimc[1], sizeof(int *));
        for (i = 0; i < dimc[1]; i++)
            csplit[i] = &(csplit2[i * dimc[0]]);
    }
    xmiss = (const int **) ALLOC((int) dimx[1], sizeof(int *));
    xdata = (const double **) ALLOC((int) dimx[1], sizeof(double *));
    for (i = 0; i < dimx[1]; i++) {
        xmiss[i] = &(xmiss2[i * dimx[0]]);
        xdata[i] = &(xdata2[i * dimx[0]]);
    }
    

    for (i = 0; i < n; i++) {
        node = 1;               /* current node of the tree */
next:
        for (npos = 0; nnum[npos] != node; npos++);  /* position of the node */

        n1[npos]++;
        wt1[npos] += wt2[i];
        trs[npos] += wt2[i] * treatment2[i];
        cons[npos] += wt2[i] * (1 - treatment2[i]);
        trsums[npos] += wt2[i] * treatment2[i] * y2[i];
        consums[npos] += wt2[i] * (1 - treatment2[i]) * y2[i];
        trsqrsums[npos] +=  wt2[i] * treatment2[i] * y2[i] * y2[i];
        consqrsums[npos] += wt2[i] * (1 - treatment2[i]) * y2[i] * y2[i];
        x_sum[npos] += IV2[i];
        y_sum[npos] += treatment2[i];
        z_sum[npos] += y2[i];
        xy_sum[npos] += IV2[i] * treatment2[i];
        xz_sum[npos] += IV2[i] * y2[i];
        yz_sum[npos] += treatment2[i] * y2[i];
        xx_sum[npos] += IV2[i] * IV2[i];
        yy_sum[npos] += treatment2[i] * treatment2[i];
        zz_sum[npos] += y2[i] * y2[i];
        x1x1_sum[npos] += 1 * 1;
        x1x2_sum[npos] += 1 * treatment2[i];
        x1x3_sum[npos] += 1 * IV2[i];   
        x1x4_sum[npos] += 1 * IV2[i] * treatment2[i]; 
        x2x1_sum[npos] += treatment2[i] * 1;
        x2x2_sum[npos] += treatment2[i] * treatment2[i];
        x2x3_sum[npos] += treatment2[i] * IV2[i];   
        x2x4_sum[npos] += treatment2[i] * IV2[i] * treatment2[i]; 
        x3x1_sum[npos] += IV2[i] * 1;
        x3x2_sum[npos] += IV2[i] * treatment2[i];
        x3x3_sum[npos] += IV2[i] * IV2[i];   
        x3x4_sum[npos] += IV2[i] * IV2[i] * treatment2[i];  
        x4x1_sum[npos] += IV2[i] * treatment2[i] * 1; 
        x4x2_sum[npos] += IV2[i] * treatment2[i] * treatment2[i];
        x4x3_sum[npos] += IV2[i] * treatment2[i] * IV2[i];   
        x4x4_sum[npos] += IV2[i] * treatment2[i] * IV2[i] * treatment2[i];  
        x1y_sum[npos] += y2[i];
        x2y_sum[npos] += y2[i] * treatment2[i];
        x3y_sum[npos] += y2[i] * IV2[i];  
        x4y_sum[npos] += y2[i] * IV2[i] * treatment2[i]; 
        /* walk down the tree */
        nspl = nodes[2][npos] - 1;      /* index of primary split */
        if (nspl >= 0) {        /* not a leaf node */
            var = vnum[nspl] - 1;
            if (xmiss[var][i] == 0) {   /* primary var not missing */
                ncat = (int) split[1][nspl];
                temp = split[3][nspl];
                if (ncat >= 2)
                    dir = csplit[(int) xdata[var][i] - 1][(int) temp - 1];
                else if (xdata[var][i] < temp)
                    dir = ncat;
                else
                    dir = -ncat;
                if (dir) {
                    if (dir == -1)
                        node = 2 * node;
                    else
                        node = 2 * node + 1;
                    goto next;
                }
            }
            if (*usesur > 0) {
                for (j = 0; j < nodes[1][npos]; j++) {
                    nspl = nodes[0][npos] + nodes[2][npos] + j;
                    var = vnum[nspl] - 1;
                    if (xmiss[var][i] == 0) {   /* surrogate not missing */
                        ncat = (int) split[1][nspl];
                        temp = split[3][nspl];
                        if (ncat >= 2)
                            dir = csplit[(int)xdata[var][i] - 1][(int)temp - 1];
                        else if (xdata[var][i] < temp)
                            dir = ncat;
                        else
                            dir = -ncat;
                        if (dir) {
                            if (dir == -1)
                                node = 2 * node;
                            else
                                node = 2 * node + 1;
                            goto next;
                        }
                    }
                }
            }
            if (*usesur > 1) {  /* go with the majority */
                for (j = 0; nnum[j] != (2 * node); j++);
                lcount = n1[j];
                for (j = 0; nnum[j] != (1 + 2 * node); j++);
                rcount = n1[j];
                if (lcount != rcount) {
                    if (lcount > rcount)
                        node = 2 * node;
                    else
                        node = 2 * node + 1;
                    goto next;
                }
            }
        }
        where[i] = node;
    }
    
    for (i = 0; i <= nnodemax; i++) {
        if (invertdx[i] == -1)
            continue;
        int origindx = invertdx[i];
        //base case
        if (trs[origindx] != 0 && cons[origindx] != 0) {
            //double tr_mean = trsums[origindx] * 1.0 / trs[origindx];
            //double con_mean = consums[origindx] * 1.0 / cons[origindx];
            //yval1[origindx] = tr_mean - con_mean;            
            //dev1[origindx] = trsqrsums[origindx] - trs[origindx] * tr_mean * tr_mean 
            //    + consqrsums[origindx] - cons[origindx] * con_mean * con_mean;     
    m[0] = x1x1_sum[origindx];
    m[1] = x1x2_sum[origindx];
    m[2] = x1x3_sum[origindx];
    m[3] = x1x4_sum[origindx];
    m[4] = x2x1_sum[origindx];
    m[5] = x2x2_sum[origindx];
    m[6] = x2x3_sum[origindx];
    m[7] = x2x4_sum[origindx];
    m[8] = x3x1_sum[origindx];
    m[9] = x3x2_sum[origindx];
    m[10] = x3x3_sum[origindx];
    m[11] = x3x4_sum[origindx];
    m[12] = x4x1_sum[origindx];
    m[13] = x4x2_sum[origindx];
    m[14] = x4x3_sum[origindx];     
    m[15] = x4x4_sum[origindx];   
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

    for (i = 0; i < 16; i++){
        invOut[i] = inv[i] / det;
    }
    bhat_0 = invOut[0] * x1y_sum[origindx] + invOut[1] * x2y_sum[origindx] + invOut[2] * x3y_sum[origindx] + invOut[3] * x4y_sum[origindx];
    bhat_1 = invOut[4] * x1y_sum[origindx] + invOut[5] * x2y_sum[origindx] + invOut[6] * x3y_sum[origindx] + invOut[7] * x4y_sum[origindx];
    bhat_2 = invOut[8] * x1y_sum[origindx] + invOut[9] * x2y_sum[origindx] + invOut[10] * x3y_sum[origindx] + invOut[11] * x4y_sum[origindx];
    bhat_3 = invOut[12] * x1y_sum[origindx] + invOut[13] * x2y_sum[origindx] + invOut[14] * x3y_sum[origindx] + invOut[15] * x4y_sum[origindx];

//    for (i = 0; i < n; i++) {
//        error2 += (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) * (*y[i] - bhat_0 - bhat_1 * treatment[i] - bhat_2 * IV[i] - bhat_3 * IV[i] * treatment[i]) / (n - 4); 
//    }
    error2 = bhat_0*bhat_0 + 2*bhat_0*bhat_1*x1x2_sum[origindx] + 2*bhat_0*bhat_2*x1x3_sum[origindx] + 2*bhat_0*bhat_3*x1x4_sum[origindx] - 2*bhat_0*x1y_sum[origindx] + bhat_1*bhat_1*x2x2_sum[origindx] + 2*bhat_1*bhat_2*x2x3_sum[origindx] + 2*bhat_1*bhat_3*x2x4_sum[origindx] - 2*bhat_1*x2y_sum[origindx] + bhat_2*bhat_2*x3x3_sum[origindx] + 2*bhat_2*bhat_3*x3x4_sum[origindx] - 2*bhat_2*x3y_sum[origindx] + bhat_3*bhat_3*x4x4_sum[origindx] - 2*bhat_3*x4y_sum[origindx] + yy_sum[origindx];
           
    var3 = error2 * invOut[15];   
    } else {
    bhat_3 = 0;
    var3 = 0;
    }   
         
           // double alpha_1 = (n1[origindx] * xz_sum[origindx] - x_sum[origindx] * z_sum[origindx]) / (n1[origindx] * xy_sum[origindx] - x_sum[origindx] * y_sum[origindx]);
           // double alpha_0 = (z_sum[origindx] - alpha_1 * y_sum[origindx]) / n1[origindx];
           // double beta_1 = (n1[origindx] * xy_sum[origindx] - x_sum[origindx] * y_sum[origindx]) / (n1[origindx] * xx_sum[origindx] - x_sum[origindx] * x_sum[origindx]);
           // double beta_0 = (y_sum[origindx] - beta_1 * x_sum[origindx]) / n1[origindx];
           // yval1[origindx] = alpha_1;
              yval1[origindx] = bhat_3;
           // double numerator = zz_sum[origindx] + n1[origindx] * alpha_0 * alpha_0 + alpha_1 * alpha_1 * yy_sum[origindx] - 2 * alpha_0 * z_sum[origindx] - 2 * alpha_1 * yz_sum[origindx] + 2 * alpha_0 * alpha_1 * y_sum[origindx];
           // double denominator = n1[origindx] * beta_0 * beta_0 + beta_1 * beta_1 * xx_sum[origindx] + y_sum[origindx] * y_sum[origindx] / n1[origindx] + 2 * beta_0 * beta_1 * x_sum[origindx] - 2 * beta_0 * y_sum[origindx] - 2 * beta_1 * x_sum[origindx] * y_sum[origindx] / n1[origindx];
           // dev1[origindx] = numerator / denominator;
              dev1[origindx] = var3;
        } else {
            int parentdx = invertdx[i / 2];
            yval1[origindx] = yval1[parentdx];
            dev1[origindx] = yval1[parentdx];
        }
    }
    
}
   
#include <Rinternals.h>

SEXP
honest_estimate_causalTree(SEXP dimx, SEXP nnode, 
                           SEXP nsplit, SEXP dimc, SEXP nnum, 
                           SEXP nodes2, 
                           SEXP n1, SEXP wt1, SEXP dev1, SEXP yval1, 
                           SEXP vnum, 
                           SEXP split2,
                           SEXP csplit2, SEXP usesur, 
                           SEXP xdata2, SEXP wt2, SEXP treatment2, SEXP IV2, SEXP y2,
                           SEXP xmiss2)
{
    int n = asInteger(dimx);
    SEXP where = PROTECT(allocVector(INTSXP, n));
    honest_estimate_causalTree0(INTEGER(dimx), asInteger(nnode), asInteger(nsplit),
            INTEGER(dimc), INTEGER(nnum), INTEGER(nodes2),
            INTEGER(vnum), REAL(split2), INTEGER(csplit2),
            INTEGER(usesur), 
            INTEGER(n1), REAL(wt1), REAL(dev1), REAL(yval1), 
            REAL(xdata2), REAL(wt2), REAL(treatment2), REAL(IV2), REAL(y2),
            INTEGER(xmiss2), INTEGER(where));
    UNPROTECT(1);
    return where;
}
