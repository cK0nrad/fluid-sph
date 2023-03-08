use std::{f64::EPSILON, ops::Add};
//Copy from java lib
pub fn tql2(v: &mut [[f64; 3]; 3], e: &mut [f64; 3], d: &mut [f64; 3]) {
    for i in 1..3 {
        e[i - 1] = e[i];
    }
    e[2] = 0.0;

    let mut f: f64 = 0.0;
    let mut tst1: f64 = 0.0;
    let eps: f64 = 2.0_f64.powi(-52);
    for l in 0..3 {
        tst1 = tst1.max(d[l].abs().add(e[l].abs()));
        let mut m = l;
        while m < 3 {
            if e[m].abs() <= eps * tst1 {
                break;
            }
            m += 1;
        }

        if m > l {
            let mut iter = 0;
            loop {
                iter = iter + 1; // (Could check iteration count here.)

                // Compute implicit shift

                let mut g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = p.powi(2).add(1.0_f64).sqrt();
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for i in l + 2..3 {
                    d[i] -= h;
                }
                f = f + h;

                // Implicit QL transformation.

                p = d[m];
                let mut c = 1.0;
                let mut c2 = c;
                let mut c3 = c;
                let el1 = e[l + 1];
                let mut s = 0.0;
                let mut s2 = 0.0;

                let mut iz = (m - 1) as i64;
                while iz >= l as i64 {
                    let i = iz as usize;
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = p.powi(2).add(e[i].powi(2)).sqrt();
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);

                    // Accumulate transformation.

                    for k in 0..3 {
                        h = v[k][i + 1];
                        v[k][i + 1] = s * v[k][i] + c * h;
                        v[k][i] = c * v[k][i] - s * h;
                    }
                    iz -= 1;
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;

                // Check for convergence.
                if !(e[l].abs() > eps * tst1) {
                    break;
                }
            }
        }
        d[l] = d[l] + f;
        e[l] = 0.0;
    }

    // Sort eigenvalues and corresponding vectors.

    for i in 0..2 {
        let mut k = i;
        let mut p = d[i];
        for j in i + 1..3 {
            if d[j] < p {
                k = j;
                p = d[j];
            }
        }
        if k != i {
            d[k] = d[i];
            d[i] = p;
            for j in 0..3 {
                p = v[j][i];
                v[j][i] = v[j][k];
                v[j][k] = p;
            }
        }
    }
}

pub fn householder(v: &mut [[f64; 3]; 3], e: &mut [f64; 3], d: &mut [f64; 3]) {
    for j in 0..3 {
        d[j] = v[2][j];
    }

    for i in (1..3).rev() {
        let mut scale = 0.0;
        let mut h = 0.0;

        for k in 0..i {
            scale += d[k].abs();
        }

        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[i - 1][j];
                v[i][j] = 0.0;
                v[j][i] = 0.0;
            }
        } else {
            for k in 0..i {
                d[k] /= scale;
                h += d[k].powi(2);
            }

            let mut f = d[i - 1];
            let mut g = h.sqrt();

            if f > 0.0 {
                g = -g;
            }

            e[i] = scale * g;
            h -= f * g;

            d[i - 1] = f - g;
            for j in 0..i {
                e[j] = 0.0;
            }

            for j in 0..i {
                f = d[j];
                v[j][i] = f;
                g = e[j] + v[j][j] * f;
                for k in j + 1..i {
                    g += v[k][j] * d[k];
                    e[k] += v[k][j] * f;
                }
                e[j] = g;
            }

            f = 0.0;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }

            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }

            for j in 0..i {
                f = d[j];
                g = e[j];
                for k in j..=i - 1 {
                    v[k][j] -= f * e[k] + g * d[k] + EPSILON;
                }
                d[j] = v[i - 1][j];
                v[i][j] = 0.0;
            }
        }
        d[i] = h;
    }

    for i in 0..2 {
        v[2][i] = v[i][i];
        v[i][i] = 1.0;
        let h = d[i + 1];

        if h != 0.0 {
          
          for k in 0..=i {
            d[k] = v[k][i + 1] / h;
          }

          for j in 0..=i {
                let mut g = 0.0;
                for k in 0..=i {
                    g += v[k][i + 1] * v[k][j];
                }
                for k in 0..=i {
                    v[k][j] -= g * d[k];
                }
          }
        }
        for k in 0..=i {
            v[k][i + 1] = 0.0;
        }
    }
    for j in 0..3 {
        d[j] = v[2][j];
        v[2][j] = 0.0;
    }
    v[2][2] = 1.0;
    e[0] = 0.0;
}

pub fn eigen(matrix: [[f64; 3]; 3]) -> ([[f64; 3]; 3], [f64; 3]) {
    let mut e = [0.0; 3];

    let mut v = [[0.0; 3]; 3];
    let mut d = [0.0; 3];

    for l in 0..3 {
        for m in 0..3 {
            v[l][m] = matrix[l][m];
        }
    }

    householder(&mut v, &mut e, &mut d);
    tql2(&mut v, &mut e, &mut d);
    return (v, d);
}
