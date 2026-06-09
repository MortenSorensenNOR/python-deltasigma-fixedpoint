/*
 * _simulateDSM_fpcore.c
 * CPython C extension: fixed-point inner loop for simulateDSM.
 * This file is part of python-deltasigma.
 *
 * Copyright (c) 2026, Morten Sørensen
 * SPDX-License-Identifier: BSD-2-Clause
 *
 * Implements the same state-space simulation loop as _simulateDSM_fixedpoint.py
 * but entirely in C using int64 fixed-point arithmetic.
 *
 * Accumulation model (mirrors the Python fixedpoint library exactly):
 *   Each matrix-vector product (C*x, D1*u, A*x, B*[u;v]) is computed as a
 *   separate inner accumulation, matching _simulateDSM_fixedpoint._matvec_fp.
 *   Within each accumulation, the running acc is extended to product precision
 *   before adding the new term, and the combined sum is rounded back to target
 *   in one step -- identical to `acc = acc + term; acc.resize(m, n)`.  The
 *   Cx and D1u results are then added and resized, mirroring the Python loop
 *   `s = Cx_fp[r] + D1u_fp[r]; s.resize(...)`.
 *
 * Requirements: GCC or Clang (for __int128), 64-bit platform.
 * Total format width (m+n) must not exceed 62 bits per signal class.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Constants (must match the Python wrapper's encoding)
 * ------------------------------------------------------------------------- */
#define ROUND_CONVERGENT  0
#define ROUND_NEAREST     1
#define ROUND_DOWN        2
#define ROUND_UP          3

#define OVF_CLAMP   0
#define OVF_WRAP    1

#define ALERT_ERROR   0
#define ALERT_WARNING 1
#define ALERT_IGNORE  2

/* -------------------------------------------------------------------------
 * Q-format descriptor
 * ------------------------------------------------------------------------- */
typedef struct {
    int     signed_;
    int     m;
    int     n;
    int     overflow;
    int     rounding;
    int     overflow_alert;
    int     total_bits;
    int64_t min_val;
    int64_t max_val;
} QFmt;

static int qfmt_init(QFmt *q, int signed_, int m, int n,
                     int overflow, int rounding, int overflow_alert)
{
    q->signed_        = signed_;
    q->m              = m;
    q->n              = n;
    q->overflow       = overflow;
    q->rounding       = rounding;
    q->overflow_alert = overflow_alert;
    q->total_bits     = m + n;

    if (q->total_bits > 62) {
        PyErr_Format(PyExc_ValueError,
            "QFmt: m+n=%d exceeds the 62-bit limit of the C extension; "
            "reduce word length or use the pure-Python backend.",
            q->total_bits);
        return -1;
    }
    if (signed_) {
        q->min_val = -(INT64_C(1) << (m + n - 1));
        q->max_val =  (INT64_C(1) << (m + n - 1)) - 1;
    } else {
        q->min_val = 0;
        q->max_val = (INT64_C(1) << (m + n)) - 1;
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * apply_overflow128: clamp/wrap an __int128 to q's range, write to int64.
 * Returns 0 on success, -1 with exception set on error.
 * ------------------------------------------------------------------------- */
static int apply_overflow128(__int128 val, const QFmt *q, int64_t *out)
{
    __int128 lo = (__int128)q->min_val;
    __int128 hi = (__int128)q->max_val;

    if (val >= lo && val <= hi) {
        *out = (int64_t)val;
        return 0;
    }
    if (q->overflow_alert == ALERT_ERROR) {
        PyErr_Format(PyExc_OverflowError,
            "Fixed-point overflow: value out of range [%lld, %lld] "
            "(Q%d.%d %s)",
            (long long)q->min_val, (long long)q->max_val,
            q->m, q->n, q->signed_ ? "signed" : "unsigned");
        return -1;
    }
    if (q->overflow == OVF_CLAMP) {
        *out = (val < lo) ? q->min_val : q->max_val;
    } else {
        __int128 span    = (__int128)1 << q->total_bits;
        __int128 shifted = val - lo;
        __int128 wrapped = ((shifted % span) + span) % span;
        *out = (int64_t)(wrapped + lo);
    }
    return 0;
}

static int apply_overflow(int64_t val, const QFmt *q, int64_t *out)
{
    return apply_overflow128((__int128)val, q, out);
}

/* -------------------------------------------------------------------------
 * float_to_fp: convert a double to a Q-format raw integer.
 * Uses the format's own rounding mode.
 * ------------------------------------------------------------------------- */
static int float_to_fp(double val, const QFmt *q, int64_t *out)
{
    double scaled  = val * ldexp(1.0, q->n);
    double floored = floor(scaled);
    double diff    = scaled - floored;
    int64_t raw;

    switch (q->rounding) {
        case ROUND_DOWN:
            raw = (int64_t)floored;
            break;
        case ROUND_UP:
            raw = (diff > 0.0) ? (int64_t)(floored + 1.0) : (int64_t)floored;
            break;
        case ROUND_NEAREST:
            raw = (diff >= 0.5) ? (int64_t)(floored + 1.0) : (int64_t)floored;
            break;
        case ROUND_CONVERGENT:
        default:
            if (diff > 0.5)
                raw = (int64_t)(floored + 1.0);
            else if (diff < 0.5)
                raw = (int64_t)floored;
            else {
                int64_t f = (int64_t)floored;
                raw = (f & 1) ? f + 1 : f;
            }
            break;
    }
    return apply_overflow(raw, q, out);
}

/* -------------------------------------------------------------------------
 * fp_mul_acc: multiply a (na frac-bits) by b (nb frac-bits), extend acc
 * (target format) to product precision, compute the combined sum, then
 * round and clip the total back to target.
 *
 * This mirrors Python's `acc = acc + term; acc.resize(m, n)` exactly:
 *   - acc is extended to product precision (left-shift by shift bits)
 *   - the product is added to extended acc (no intermediate rounding)
 *   - the combined sum is rounded back to target in a single step
 *
 * Returns 0 on success, -1 if an overflow error must propagate.
 * ------------------------------------------------------------------------- */
static int fp_mul_acc(int64_t a, int na,
                      int64_t b, int nb,
                      int64_t acc,
                      const QFmt *qt,
                      int64_t *out_acc)
{
    __int128 product = (__int128)a * (__int128)b;
    int prod_n = na + nb;
    int shift  = prod_n - qt->n;   /* >0: right-shift to target; <0: left-shift */

    __int128 result;

    if (shift > 0) {
        /* Extend acc to product precision, sum, then round back to target. */
        __int128 acc_ext = (__int128)acc << shift;
        __int128 sum     = product + acc_ext;

        __int128 quotient = sum >> shift;          /* arithmetic right-shift */
        __int128 r        = sum - (quotient << shift);
        __int128 half     = (__int128)1 << (shift - 1);

        switch (qt->rounding) {
            case ROUND_DOWN:
                result = quotient; break;
            case ROUND_UP:
                result = quotient + (r > 0 ? 1 : 0); break;
            case ROUND_NEAREST:
                result = quotient + (r >= half ? 1 : 0); break;
            case ROUND_CONVERGENT:
            default:
                if (r > half)
                    result = quotient + 1;
                else if (r < half)
                    result = quotient;
                else
                    result = quotient + ((quotient & 1) ? 1 : 0);
                break;
        }
    } else if (shift == 0) {
        /* Formats already aligned: just add. */
        result = product + (__int128)acc;
    } else {
        /* Product has fewer frac bits than target: left-shift product. */
        result = (product << (-shift)) + (__int128)acc;
    }

    return apply_overflow128(result, qt, out_acc);
}

/* -------------------------------------------------------------------------
 * add_fp: add two values already in the same Q-format, apply overflow.
 * Mirrors Python's `s = a + b; s.resize(m, n)` for same-format operands.
 * ------------------------------------------------------------------------- */
static int add_fp(int64_t a, int64_t b, const QFmt *q, int64_t *out)
{
    __int128 sum = (__int128)a + (__int128)b;
    return apply_overflow128(sum, q, out);
}

/* -------------------------------------------------------------------------
 * matvec_fp: compute one row of M (coeff format nc) times vec (state format
 * ns) and accumulate into a row output in target format qt.
 * Mirrors _matvec_fp in _simulateDSM_fixedpoint.py.
 * ------------------------------------------------------------------------- */
static int matvec_row(const int64_t *M_row, int nc,
                      const int64_t *vec, int ns,
                      int n_cols,
                      const QFmt *qt,
                      int64_t *out)
{
    int64_t acc = 0;
    for (int c = 0; c < n_cols; c++) {
        if (fp_mul_acc(M_row[c], nc, vec[c], ns, acc, qt, &acc) < 0)
            return -1;
    }
    *out = acc;
    return 0;
}

/* -------------------------------------------------------------------------
 * ds_quantize_single: single-sample quantizer.
 *   nlev even  → mid-rise:  v = 2*floor(y/2)   + 1
 *   nlev odd   → mid-tread: v = 2*floor((y+1)/2)
 * Output clamped to [-(nlev-1), nlev-1].
 * ------------------------------------------------------------------------- */
static double ds_quantize_single(double y, int nlev)
{
    double v;
    if (nlev % 2 == 0)
        v = 2.0 * floor(0.5 * y) + 1.0;
    else
        v = 2.0 * floor(0.5 * (y + 1.0));

    double L = (double)(nlev - 1);
    if (v >  L) v =  L;
    if (v < -L) v = -L;
    return v;
}

/* -------------------------------------------------------------------------
 * simulate_fp_inner(A, B, C, D1, order, nu, nq, u, x0, nlev,
 *                   ms,ns,sgns,ovfs,rnds,alerts,
 *                   mc,nc,sgnc,ovfc,rndc,alertc,
 *                   mi,ni,sgni,ovfi,rndi,alerti,
 *                   my,ny,sgny,ovfy,rndy,alerty)
 *
 * All matrix arguments are float64 1-D arrays in C (row-major) order.
 * Returns (v, xn, xmax, y) as float64 ndarrays:
 *   (nq, N), (order, N), (order, 1), (nq, N).
 * The Python wrapper squeezes these before returning to the caller.
 * ------------------------------------------------------------------------- */
static PyObject *py_simulate_fp_inner(PyObject *self, PyObject *args)
{
    PyObject *A_obj, *B_obj, *C_obj, *D1_obj;
    PyObject *u_obj, *x0_obj, *nlev_obj;
    int order, nu, nq;
    int ms,ns,sgns,ovfs,rnds,alerts;
    int mc,nc,sgnc,ovfc,rndc,alertc;
    int mi,ni,sgni,ovfi,rndi,alerti;
    int my,ny,sgny,ovfy,rndy,alerty;

    if (!PyArg_ParseTuple(args,
            "OOOOiiiOOO"
            "iiiiii"
            "iiiiii"
            "iiiiii"
            "iiiiii",
            &A_obj, &B_obj, &C_obj, &D1_obj,
            &order, &nu, &nq,
            &u_obj, &x0_obj, &nlev_obj,
            &ms, &ns, &sgns, &ovfs, &rnds, &alerts,
            &mc, &nc, &sgnc, &ovfc, &rndc, &alertc,
            &mi, &ni, &sgni, &ovfi, &rndi, &alerti,
            &my, &ny, &sgny, &ovfy, &rndy, &alerty))
        return NULL;

    QFmt q_state, q_coeff, q_input, q_y;
    if (qfmt_init(&q_state, sgns, ms, ns, ovfs, rnds, alerts)  != 0) return NULL;
    if (qfmt_init(&q_coeff, sgnc, mc, nc, ovfc, rndc, alertc)  != 0) return NULL;
    if (qfmt_init(&q_input, sgni, mi, ni, ovfi, rndi, alerti)  != 0) return NULL;
    if (qfmt_init(&q_y,     sgny, my, ny, ovfy, rndy, alerty)  != 0) return NULL;

    PyArrayObject *A_arr  = (PyArrayObject*)PyArray_FROM_OTF(A_obj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *B_arr  = (PyArrayObject*)PyArray_FROM_OTF(B_obj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *C_arr  = (PyArrayObject*)PyArray_FROM_OTF(C_obj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *D1_arr = (PyArrayObject*)PyArray_FROM_OTF(D1_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *u_arr  = (PyArrayObject*)PyArray_FROM_OTF(u_obj,  NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *x0_arr = (PyArrayObject*)PyArray_FROM_OTF(x0_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *nlev_arr=(PyArrayObject*)PyArray_FROM_OTF(nlev_obj,NPY_INT64,  NPY_ARRAY_IN_ARRAY);

    PyArrayObject *v_arr    = NULL;
    PyArrayObject *xn_arr   = NULL;
    PyArrayObject *xmax_arr = NULL;
    PyArrayObject *y_arr    = NULL;

    int64_t *A_fp    = NULL;
    int64_t *B_fp    = NULL;
    int64_t *C_fp    = NULL;
    int64_t *D1_fp   = NULL;
    int64_t *x_fp    = NULL;
    int64_t *x_next  = NULL;
    int64_t *u_fp    = NULL;
    int64_t *v_fp    = NULL;
    int64_t *Cx_row  = NULL;   /* C*x result per quantizer row */
    int64_t *D1u_row = NULL;   /* D1*u result per quantizer row */
    int64_t *Ax_row  = NULL;   /* A*x result per state row */
    int64_t *Buv_row = NULL;   /* B*[u;v] result per state row */
    double  *y_tmp   = NULL;

    PyObject *result = NULL;

    if (!A_arr || !B_arr || !C_arr || !D1_arr || !u_arr || !x0_arr || !nlev_arr)
        goto done;

    {
        double  *A    = (double*)PyArray_DATA(A_arr);
        double  *B    = (double*)PyArray_DATA(B_arr);
        double  *C    = (double*)PyArray_DATA(C_arr);
        double  *D1   = (double*)PyArray_DATA(D1_arr);
        double  *u    = (double*)PyArray_DATA(u_arr);
        double  *x0   = (double*)PyArray_DATA(x0_arr);
        int64_t *nlev = (int64_t*)PyArray_DATA(nlev_arr);

        int N = (int)((int)PyArray_SIZE(u_arr) / nu);

        npy_intp dims_vxy[2]  = {nq, N};
        npy_intp dims_xn[2]   = {order, N};
        npy_intp dims_xmax[2] = {order, 1};

        v_arr    = (PyArrayObject*)PyArray_ZEROS(2, dims_vxy,  NPY_DOUBLE, 0);
        xn_arr   = (PyArrayObject*)PyArray_ZEROS(2, dims_xn,   NPY_DOUBLE, 0);
        xmax_arr = (PyArrayObject*)PyArray_ZEROS(2, dims_xmax, NPY_DOUBLE, 0);
        y_arr    = (PyArrayObject*)PyArray_ZEROS(2, dims_vxy,  NPY_DOUBLE, 0);
        if (!v_arr || !xn_arr || !xmax_arr || !y_arr) goto done;

        double *v_data    = (double*)PyArray_DATA(v_arr);
        double *xn_data   = (double*)PyArray_DATA(xn_arr);
        double *xmax_data = (double*)PyArray_DATA(xmax_arr);
        double *y_data    = (double*)PyArray_DATA(y_arr);

        int A_sz  = order * order;
        int B_sz  = order * (nu + nq);
        int C_sz  = nq * order;
        int D1_sz = nq * nu;

        A_fp    = (int64_t*)malloc(A_sz  * sizeof(int64_t));
        B_fp    = (int64_t*)malloc(B_sz  * sizeof(int64_t));
        C_fp    = (int64_t*)malloc(C_sz  * sizeof(int64_t));
        D1_fp   = (int64_t*)malloc(D1_sz * sizeof(int64_t));
        x_fp    = (int64_t*)malloc(order * sizeof(int64_t));
        x_next  = (int64_t*)malloc(order * sizeof(int64_t));
        u_fp    = (int64_t*)malloc(nu    * sizeof(int64_t));
        v_fp    = (int64_t*)malloc(nq    * sizeof(int64_t));
        Cx_row  = (int64_t*)malloc(nq    * sizeof(int64_t));
        D1u_row = (int64_t*)malloc(nq    * sizeof(int64_t));
        Ax_row  = (int64_t*)malloc(order * sizeof(int64_t));
        Buv_row = (int64_t*)malloc(order * sizeof(int64_t));
        y_tmp   = (double* )malloc(nq    * sizeof(double));

        if (!A_fp || !B_fp || !C_fp || !D1_fp || !x_fp || !x_next ||
            !u_fp || !v_fp || !Cx_row || !D1u_row || !Ax_row || !Buv_row || !y_tmp) {
            PyErr_NoMemory();
            goto done;
        }

        for (int i = 0; i < A_sz;  i++) if (float_to_fp(A[i],  &q_coeff, &A_fp[i])  < 0) goto done;
        for (int i = 0; i < B_sz;  i++) if (float_to_fp(B[i],  &q_coeff, &B_fp[i])  < 0) goto done;
        for (int i = 0; i < C_sz;  i++) if (float_to_fp(C[i],  &q_coeff, &C_fp[i])  < 0) goto done;
        for (int i = 0; i < D1_sz; i++) if (float_to_fp(D1[i], &q_coeff, &D1_fp[i]) < 0) goto done;

        for (int r = 0; r < order; r++) {
            if (float_to_fp(x0[r], &q_state, &x_fp[r]) < 0) goto done;
            xmax_data[r] = fabs(x0[r]);
        }

        /* ---- Main simulation loop ---- */
        for (int i = 0; i < N; i++) {

            /* Input column → fixed-point */
            for (int r = 0; r < nu; r++) {
                if (float_to_fp(u[r * N + i], &q_input, &u_fp[r]) < 0) goto done;
            }

            /* ---- Pass 1: Cx_row[q] = C[q,:] @ x  (in q_y format) ---- */
            for (int q = 0; q < nq; q++) {
                if (matvec_row(&C_fp[q * order], q_coeff.n,
                               x_fp, q_state.n, order,
                               &q_y, &Cx_row[q]) < 0) goto done;
            }

            /* ---- Pass 2: D1u_row[q] = D1[q,:] @ u  (in q_y format) ---- */
            for (int q = 0; q < nq; q++) {
                if (matvec_row(&D1_fp[q * nu], q_coeff.n,
                               u_fp, q_input.n, nu,
                               &q_y, &D1u_row[q]) < 0) goto done;
            }

            /* ---- y[q] = clip(Cx[q] + D1u[q])  (mirrors s=Cx+D1u; s.resize) ---- */
            for (int q = 0; q < nq; q++) {
                int64_t y_raw;
                if (add_fp(Cx_row[q], D1u_row[q], &q_y, &y_raw) < 0) goto done;
                double y_float    = (double)y_raw * ldexp(1.0, -q_y.n);
                y_tmp[q]          = y_float;
                y_data[q * N + i] = y_float;

                /* Quantize */
                int    nlev_q     = (int)nlev[nq == 1 ? 0 : q];
                double v_val      = ds_quantize_single(y_float, nlev_q);
                v_data[q * N + i] = v_val;

                if (float_to_fp(v_val, &q_coeff, &v_fp[q]) < 0) goto done;
            }

            /* ---- Pass 3: Ax_row[r] = A[r,:] @ x  (in q_state format) ---- */
            for (int r = 0; r < order; r++) {
                if (matvec_row(&A_fp[r * order], q_coeff.n,
                               x_fp, q_state.n, order,
                               &q_state, &Ax_row[r]) < 0) goto done;
            }

            /* ---- Pass 4: Buv_row[r] = B[r,:] @ [u; v]  (in q_state format) ---- */
            for (int r = 0; r < order; r++) {
                int64_t acc = 0;
                /* B columns for u (first nu columns) */
                for (int c = 0; c < nu; c++) {
                    if (fp_mul_acc(B_fp[r * (nu + nq) + c], q_coeff.n,
                                   u_fp[c], q_input.n,
                                   acc, &q_state, &acc) < 0) goto done;
                }
                /* B columns for v (next nq columns) */
                for (int c = 0; c < nq; c++) {
                    if (fp_mul_acc(B_fp[r * (nu + nq) + nu + c], q_coeff.n,
                                   v_fp[c], q_coeff.n,
                                   acc, &q_state, &acc) < 0) goto done;
                }
                Buv_row[r] = acc;
            }

            /* ---- x_next[r] = clip(Ax[r] + Buv[r])  (mirrors s=Ax+Buv; s.resize) ---- */
            for (int r = 0; r < order; r++) {
                if (add_fp(Ax_row[r], Buv_row[r], &q_state, &x_next[r]) < 0) goto done;
            }

            /* Commit x_next → x_fp; record state and update xmax */
            for (int r = 0; r < order; r++) {
                x_fp[r]            = x_next[r];
                double x_float     = (double)x_fp[r] * ldexp(1.0, -q_state.n);
                xn_data[r * N + i] = x_float;
                double ax = fabs(x_float);
                if (ax > xmax_data[r]) xmax_data[r] = ax;
            }
        }

        result = PyTuple_New(4);
        if (!result) goto done;
        PyTuple_SET_ITEM(result, 0, (PyObject*)v_arr);    v_arr    = NULL;
        PyTuple_SET_ITEM(result, 1, (PyObject*)xn_arr);   xn_arr   = NULL;
        PyTuple_SET_ITEM(result, 2, (PyObject*)xmax_arr); xmax_arr = NULL;
        PyTuple_SET_ITEM(result, 3, (PyObject*)y_arr);    y_arr    = NULL;
    }

done:
    free(A_fp); free(B_fp); free(C_fp); free(D1_fp);
    free(x_fp); free(x_next); free(u_fp); free(v_fp);
    free(Cx_row); free(D1u_row); free(Ax_row); free(Buv_row); free(y_tmp);
    Py_XDECREF(A_arr); Py_XDECREF(B_arr); Py_XDECREF(C_arr); Py_XDECREF(D1_arr);
    Py_XDECREF(u_arr); Py_XDECREF(x0_arr); Py_XDECREF(nlev_arr);
    Py_XDECREF(v_arr);
    Py_XDECREF(xn_arr);
    Py_XDECREF(xmax_arr);
    Py_XDECREF(y_arr);
    return result;
}

/* -------------------------------------------------------------------------
 * Module table and init
 * ------------------------------------------------------------------------- */
static PyMethodDef FpcoreMethods[] = {
    {"simulate_fp_inner", py_simulate_fp_inner, METH_VARARGS,
     "Fixed-point DSM inner loop (C implementation)."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fpcoremodule = {
    PyModuleDef_HEAD_INIT,
    "_simulateDSM_fpcore",
    NULL,
    -1,
    FpcoreMethods
};

PyMODINIT_FUNC PyInit__simulateDSM_fpcore(void)
{
    import_array();
    return PyModule_Create(&fpcoremodule);
}
