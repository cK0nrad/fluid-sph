//https://en.wikipedia.org/wiki/Marching_cubes

// Convert simulation to luxrender usable files

use std::{
    collections::{HashMap, BTreeSet},
    f64::consts::PI,
    ops::*, fs::{OpenOptions, self}, sync::Arc,
};
use std::io::Write;
use rayon::prelude::*;

use crate::{eigen_value, vectors::Vector, DensityPosition};

static EDGE_TABLE: [i64; 256] = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03,
    0xe09, 0xf00, 0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f,
    0xa96, 0xd9a, 0xc93, 0xf99, 0xe90, 0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30, 0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6,
    0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0, 0x460, 0x569,
    0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69,
    0xb60, 0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6,
    0x9fa, 0x8f3, 0xbf9, 0xaf0, 0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c, 0xe5c,
    0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950, 0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf,
    0x1c5, 0xcc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0, 0x8c0, 0x9c9, 0xac3,
    0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x55, 0x35f, 0x256, 0x55a,
    0x453, 0x759, 0x650, 0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5,
    0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0, 0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65,
    0xc6c, 0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460, 0xca0, 0xda9, 0xea3, 0xfaa,
    0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0, 0xd30,
    0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33,
    0x339, 0x230, 0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f,
    0x596, 0x29a, 0x393, 0x99, 0x190, 0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
];

static TRI_TABLE: [[i64; 16]; 256] = [
    [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    ],
    [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
    [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
    [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
    [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
    [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
    [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
    [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
    [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
    [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
    [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
    [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
    [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
    [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
    [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
    [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
    [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
    [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
    [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
    [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
    [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
    [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
    [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
    [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
    [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
    [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
    [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
    [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
    [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
    [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
    [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
    [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
    [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
    [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
    [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
    [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
    [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
    [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
    [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
    [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
    [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
    [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
    [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
    [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
    [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
    [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
    [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
    [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
    [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
    [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
    [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
    [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
    [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
    [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    ],
];
static DX: [usize; 8] = [0, 0, 1, 1, 0, 0, 1, 1];
static DY: [usize; 8] = [0, 1, 1, 0, 0, 1, 1, 0];
static DZ: [usize; 8] = [0, 0, 0, 0, 1, 1, 1, 1];

static POLY6C: f64 = 315.0_f64 / 64.0_f64 / PI;
static KSCALE: f64 = 1.0 / 2.5;
static ISOLEVEL: f64 = 0.08;

fn poly6(u2: f64) -> f64 {
    if u2 > 1.0 {
        return 0.0;
    }
    let d = 1.0 - u2;
    POLY6C.mul(d.powi(3))
}

fn vertex_interpret(
    a: String,
    b: String,
    min_vector: Vector,
    baselen: f64,
    valuemap: &mut HashMap<String, f64>,
    edgepos: &mut Vec<Vector>,
    edgemap: &mut HashMap<String, usize>,
) -> usize {
    let mut test_string = a.clone();
    test_string.push_str(&b);
    if !edgemap.contains_key(&test_string) {
        if !valuemap.contains_key(&a.clone()) {
            valuemap.insert(a.clone(), 0.0);
        }
        let v_a = valuemap[&a];

        if !valuemap.contains_key(&b.clone()) {
            valuemap.insert(b.clone(), 0.0);
        }
        let v_b = valuemap[&b];

        let vec = a.split(":").collect::<Vec<&str>>();
        let l = vec[0].parse::<usize>().unwrap();
        let m = vec[1].parse::<usize>().unwrap();
        let n = vec[2].parse::<usize>().unwrap();
        let mut p_a = Vector::new(l as f64, m as f64, n as f64);

        p_a = p_a.mulf(baselen)
            .addv(min_vector);

        let vec = b.split(":").collect::<Vec<&str>>();
        let l = vec[0].parse::<usize>().unwrap();
        let m = vec[1].parse::<usize>().unwrap();
        let n = vec[2].parse::<usize>().unwrap();
        let mut p_b = Vector::new(l as f64, m as f64, n as f64);
        p_b = p_b.mulf(baselen)
            .addv(min_vector);
            
        let p: Vector;

        if (v_a - v_b).abs() > 1e-5 {
            p = p_a.addv(p_b.subv(p_a).mulf(ISOLEVEL.sub(v_a).div(v_b - v_a)));
        } else {
            p = p_a;
        }

        edgepos.push(p);
        if edgemap.contains_key(&test_string) {
            let value = edgemap.get_mut(&test_string).unwrap();
            *value = edgepos.len() - 1;
        }else {
            edgemap.insert(test_string.clone(), edgepos.len() - 1);
        }
    }
    return edgemap[&test_string];
}

pub struct Renderer {
    particle_amount: usize,
    frame: usize,

    input: Arc<Vec<DensityPosition>>,
    h: f64,
    mass: f64,

    min_vector: Vector,
    max_vector: Vector,

    det_g: Vec<f64>,
    g: Vec<Vec<Vec<f64>>>,
    bbox: Vec<Vec<Vector>>,
    new_pos: Vec<Vector>,
    preprocess_grid: Vec<Vec<Vec<Vec<usize>>>>,
    edgemap: HashMap<String, usize>,
    valuemap: HashMap<String, f64>,
    gridset: BTreeSet<String>,
    edgepos: Vec<Vector>,
}

impl Renderer {
    pub fn new(
        particle_amount: usize,
        frame: usize,
        input: Arc<Vec<DensityPosition>>,
        h: f64,
        mass: f64,
    ) -> Self {
        Self {
            particle_amount,
            frame,
            input,
            h,
            mass,
            min_vector: Vector::new(0.0, 0.0, 0.0),
            max_vector: Vector::new(0.0, 0.0, 0.0),
            det_g: vec![0.0; particle_amount],
            g: vec![vec![vec![0.0; 3]; 3]; particle_amount],
            bbox: vec![vec![Vector::new(0.0, 0.0, 0.0); 2]; particle_amount],
            new_pos: vec![Vector::new(0.0, 0.0, 0.0); particle_amount],
            preprocess_grid: vec![vec![vec![Vec::<usize>::new(); 120]; 120]; 120],
            valuemap: HashMap::new(),
            gridset: BTreeSet::new(),
            edgemap: HashMap::new(),
            edgepos: Vec::new(),
        }
    }
    pub fn getvaluemap(&self, idx: String) -> f64 {
        if self.valuemap.contains_key(&idx) {
            return *self.valuemap.get(&idx).unwrap();
        }
        return 0.0;
    }
    pub fn weight(r: f64, h: f64) -> f64 {
        if r >= 2.0 * h {
            return 0.0;
        }
        let d = r / (2.0 * h);
        return 1.0 - d * d * d;
    }

    pub fn preprocess(&mut self, diff_vector: Vector) {
        let preprocess_grid = &mut self.preprocess_grid; //vec![vec![vec![Vec::<usize>::new(); 120];120];120];

        let length = Vector::new(
            diff_vector.get_x().div(100.0).max(2.0 * self.h),
            diff_vector.get_y().div(100.0).max(2.0 * self.h),
            diff_vector.get_z().div(100.0).max(2.0 * self.h),
        );

        
     
        for i in 0..self.particle_amount {
            let input = self.input[i + self.frame * self.particle_amount].vector;
            let grid_x = input
                .get_x()
                .sub(self.min_vector.get_x())
                .div(length.get_x())
                .ceil() as usize;
            let grid_y = input
                .get_y()
                .sub(self.min_vector.get_y())
                .div(length.get_y())
                .ceil() as usize;
            let grid_z = input
                .get_z()
                .sub(self.min_vector.get_z())
                .div(length.get_z())
                .ceil() as usize;
            preprocess_grid[grid_x][grid_y][grid_z].push(i);
            
        }

        ((0..self.particle_amount), &mut self.det_g, &mut self.bbox, &mut self.new_pos, &mut self.g).into_par_iter().for_each(|(i, det_g, bbox, new_pos_g, g) | {
            let mut neighbor = 0;
            let mut sum_wij = 0.0;
            let mut cov = [[0.0; 3]; 3];

            let mut new_pos = Vector::new(0.0, 0.0, 0.0);
            let input = self.input[i + self.frame * self.particle_amount].vector;
            let grid_x = input
                .get_x()
                .sub(self.min_vector.get_x())
                .div(length.get_x())
                .ceil() as i32;
            let grid_y = input
                .get_y()
                .sub(self.min_vector.get_y())
                .div(length.get_y())
                .ceil() as i32;
            let grid_z = input
                .get_z()
                .sub(self.min_vector.get_z())
                .div(length.get_z())
                .ceil() as i32;

            for x in 0..3_i32 {
                if x.sub(1).add(grid_x) < 0 {
                    continue;
                }
                for y in 0..3_i32 {
                    if y.sub(1).add(grid_y) < 0 {
                        continue;
                    }
                    for z in 0..3_i32 {
                        if z.sub(1).add(grid_z) < 0 {
                            continue;
                        }

                        for j in &preprocess_grid[grid_x.sub(1).add(x) as usize]
                            [grid_y.sub(1).add(y) as usize]
                            [grid_z.sub(1).add(z) as usize]
                        {
                            let j_vector = self.input[*j + self.frame * self.particle_amount].vector;
                            let r = input.subv(j_vector).square_size().sqrt();
                            let wij = Self::weight(r, self.h);
                            if wij == 0.0 {
                                continue;
                            }
                            
                            neighbor += 1;
                            sum_wij += wij;

                            new_pos = new_pos.addv(j_vector.mulf(wij));
                        }
                    }
                }
            }

            new_pos = new_pos.divf(sum_wij);

            for x in 0..3_i32 {
                if x.sub(1).add(grid_x) < 0 {
                    continue;
                }
                for y in 0..3_i32 {
                    if y.sub(1).add(grid_y) < 0 {
                        continue;
                    }
                    for z in 0..3_i32 {
                        if z.sub(1).add(grid_z) < 0 {
                            continue;
                        }

                        for j in &preprocess_grid[grid_x.sub(1).add(x) as usize]
                            [grid_y.sub(1).add(y) as usize]
                            [grid_z.sub(1).add(z) as usize]
                        {
                            let j_vector = self.input[*j + self.frame * self.particle_amount].vector;
                            let r = input.subv(j_vector).square_size().sqrt();
                            let wij = Self::weight(r, self.h);
                            if wij == 0.0 {
                                continue;
                            }

                            let delta = j_vector.subv(new_pos);
                            for l in 0..3 {
                                for m in 0..3 {
                                    cov[l as usize][m as usize] +=
                                        wij * delta.get(l) * delta.get(m);
                                }
                            }
                        }
                    }
                }
            }

            for l in 0..3 {
                for m in 0..3 {
                    cov[l as usize][m as usize] /= sum_wij;
                }
            }

            let (eiv,  mut eig) = eigen_value::eigen(cov);

            eig[0] = eig[0].max(eig[2] / 5.0);
            eig[1] = eig[1].max(eig[2] / 5.0);

            if neighbor < 35 {
                eig[0] = 0.6;
                eig[1] = 0.6;
                eig[2] = 0.6;
            } else {
                for j in 0..3 {
                    eig[j] *= KSCALE;
                }
            }

            let mut m = [[0.0_f64; 3]; 3];
            *det_g = 1.0;
            for l in 0..3 {
                for n in 0..3 {
                    m[l][n] = eiv[l][n] * eig[n] * self.h;
                    g[l][n] = eiv[n][l] / eig[l] / self.h;
                }
                *det_g *= eig[l] / self.h;
            }

            let halfbox = Vector::new(
                Vector::new(m[0][0], m[0][1], m[0][2]).square_size().sqrt(),
                Vector::new(m[1][0], m[1][1], m[1][2]).square_size().sqrt(),
                Vector::new(m[2][0], m[2][1], m[2][2]).square_size().sqrt(),
            );

            bbox[0] = new_pos.subv(halfbox);
            bbox[1] = new_pos.addv(halfbox);
            *new_pos_g = new_pos.clone();
        });
        

        for i in 0..self.particle_amount {
            let grid_x = self.input[i + self.frame * self.particle_amount]
                .vector
                .get_x()
                .sub(self.min_vector.get_x())
                .div(length.get_x())
                .ceil() as usize;
            let grid_y = self.input[i + self.frame * self.particle_amount]
                .vector
                .get_y()
                .sub(self.min_vector.get_y())
                .div(length.get_y())
                .ceil() as usize;
            let grid_z = self.input[i + self.frame * self.particle_amount]
                .vector
                .get_z()
                .sub(self.min_vector.get_z())
                .div(length.get_z())
                .ceil() as usize;
            self.preprocess_grid[grid_x][grid_y][grid_z].pop();
        }
    }

    pub fn set_frame(&mut self, frame: usize) {
        self.frame = frame;
    }

    pub fn generate(&mut self) {

        let mut min_vector = Vector::new(0.0, 0.0, 0.0);
        let mut max_vector = Vector::new(0.0, 0.0, 0.0);

        let mut matrix = Vec::<(isize, isize, isize)>::new();

        for i in 0..self.particle_amount {
            let current = self.input[i + self.frame * self.particle_amount];
            if i == 0 {
                min_vector = current.vector.clone();
                max_vector = current.vector.clone();
            }
            min_vector.set_x(min_vector.get_x().min(current.vector.get_x()));
            min_vector.set_y(min_vector.get_y().min(current.vector.get_y()));
            min_vector.set_z(min_vector.get_z().min(current.vector.get_z()));
            
            max_vector.set_x(max_vector.get_x().max(current.vector.get_x()));
            max_vector.set_y(max_vector.get_y().max(current.vector.get_y()));
            max_vector.set_z(max_vector.get_z().max(current.vector.get_z()));
        }
        
        let h_vector = Vector::new(self.h, self.h, self.h);
        self.min_vector = min_vector.subv(h_vector.mulf(4.0));
        self.max_vector = max_vector.addv(h_vector.mulf(4.0));
        min_vector = self.min_vector;
        max_vector = self.max_vector;

        let diff = max_vector.subv(min_vector);
        let baselen = 0.23;

        self.preprocess(diff);


        for i in 0..self.particle_amount {
            let bottom_left = self.bbox[i][0].subv(min_vector).divf(baselen);
            let bottom_left = Vector::new(
                bottom_left.get_x().ceil(),
                bottom_left.get_y().ceil(),
                bottom_left.get_z().ceil(),
            );

            let top_right = self.bbox[i][1].subv(min_vector).divf(baselen);
            let top_right = Vector::new(
                top_right.get_x().floor(),
                top_right.get_y().floor(),
                top_right.get_z().floor(),
            );
            for l in (bottom_left.get_x() as usize)..=(top_right.get_x() as usize) {
                for m in (bottom_left.get_y() as usize)..=(top_right.get_y() as usize) {
                    for n in (bottom_left.get_z() as usize)..=(top_right.get_z() as usize) {
                        let mut r = Vector::new(l as f64, m as f64, n as f64);
                        r = r.mulf(baselen).addv(min_vector).subv(self.new_pos[i]);
                        let mut gr = Vector::new(0.0, 0.0, 0.0);
                        for p in 0..3 {
                            for q in 0..3 {
                                let new_value = self.g[i][p][q].mul(r.get(q));
                                gr.set(p, gr.get(p).add(new_value));
                            }
                        }
                        
                        let weight = poly6(gr.square_size());

                        if weight != 0.0 {
                            let triv = format!("{}:{}:{}", l, m, n);
                            if self.valuemap.contains_key(&triv) {
                                let value = self.valuemap.get_mut(&triv).unwrap();
                                *value += self.mass / self.input[i + self.frame * self.particle_amount].density * self.det_g[i] * weight;
                            } else {
                                self.valuemap.insert(triv.clone(), self.mass / self.input[i + self.frame * self.particle_amount].density * self.det_g[i] * weight);
                            }
                        }

                    }
                }
            }

            for l in (bottom_left.get_x().sub(1.0) as isize)..=(top_right.get_x().add(1.0) as isize)
            {
                for m in (bottom_left.get_y().sub(1.0) as isize)..=(top_right.get_y().add(1.0) as isize)
                {
                    for n in (bottom_left.get_z().sub(1.0) as isize)..=(top_right.get_z().add(1.0) as isize)
                    {
                        let triv = format!("{}:{}:{}", l, m, n);
                        self.gridset.insert(triv);
                    }
                }
            }
        }

        //convert ggridset to vector

        for it in self.gridset.iter() {
            let vec = it.split(":").collect::<Vec<&str>>();
            let grid_x = vec[0].parse::<usize>().unwrap();
            let grid_y = vec[1].parse::<usize>().unwrap();
            let grid_z = vec[2].parse::<usize>().unwrap();

            let mut grid_p = vec![String::new(); 8];

            for ty in 0..8 {
                let px = grid_x + DX[ty];
                let py = grid_y + DY[ty];
                let pz = grid_z + DZ[ty];
                let triv = format!("{}:{}:{}", px, py, pz);

                grid_p[ty] = triv;
            }
            
            let mut cubeindex = 0;
            for ty in 0..8 {
                if self.getvaluemap(grid_p[ty].clone()) > ISOLEVEL {
                    cubeindex |= 1 << ty;
                }
            }

            if EDGE_TABLE[cubeindex] == 0 {
                continue;
            }

            
            let mut vertlist = [-1_isize; 12];

            let valuemap = &mut self.valuemap;
            let edgepos = &mut self.edgepos;
            let edgemap = &mut self.edgemap;
            
            

            if EDGE_TABLE[cubeindex] & 1 != 0 {
                vertlist[0] = vertex_interpret(
                    grid_p[0].clone(),
                    grid_p[1].clone(),
                    min_vector.clone(),
                    baselen,
                    valuemap,
                    edgepos,
                    edgemap,
                ) as isize;
            }
            if EDGE_TABLE[cubeindex] & 2 != 0 {
                vertlist[1] = vertex_interpret(
                    grid_p[1].clone(),
                    grid_p[2].clone(),
                    min_vector.clone(),
                    baselen,
                    valuemap,
                    edgepos,
                    edgemap,
                ) as isize;
            }
            if EDGE_TABLE[cubeindex] & 4 != 0 {
                vertlist[2] = vertex_interpret(
                    grid_p[3].clone(),
                    grid_p[2].clone(),
                    min_vector.clone(),
                    baselen,
                    valuemap,
                    edgepos,
                    edgemap,
                ) as isize;
            }
            if EDGE_TABLE[cubeindex] & 8 != 0 {
                vertlist[3] = vertex_interpret(
                    grid_p[0].clone(),
                    grid_p[3].clone(),
                    min_vector.clone(),
                    baselen,
                    valuemap,
                    edgepos,
                    edgemap,
                ) as isize;
            }
            if EDGE_TABLE[cubeindex] & 16 != 0 {
                vertlist[4] = vertex_interpret(
                    grid_p[4].clone(),
                    grid_p[5].clone(),
                    min_vector.clone(),
                    baselen,
                    valuemap,
                    edgepos,
                    edgemap,
                ) as isize;
            }
            if EDGE_TABLE[cubeindex] & 32 != 0 {
                vertlist[5] = vertex_interpret(
                    grid_p[5].clone(),
                    grid_p[6].clone(),
                    min_vector.clone(),
                    baselen,
                    valuemap,
                    edgepos,
                    edgemap,
                ) as isize;
            }
            if EDGE_TABLE[cubeindex] & 64 != 0 {
                vertlist[6] = vertex_interpret(
                    grid_p[7].clone(),
                    grid_p[6].clone(),
                    min_vector.clone(),
                    baselen,
                    valuemap,
                    edgepos,
                    edgemap,
                ) as isize;
            }
            if EDGE_TABLE[cubeindex] & 128 != 0 {
                vertlist[7] = vertex_interpret(
                    grid_p[4].clone(),
                    grid_p[7].clone(),
                    min_vector.clone(),
                    baselen,
                    valuemap,
                    edgepos,
                    edgemap,
                ) as isize;
            }
            if EDGE_TABLE[cubeindex] & 256 != 0 {
                vertlist[8] = vertex_interpret(
                    grid_p[0].clone(),
                    grid_p[4].clone(),
                    min_vector.clone(),
                    baselen,
                    valuemap,
                    edgepos,
                    edgemap,
                ) as isize;
            }
            if EDGE_TABLE[cubeindex] & 512 != 0 {
                vertlist[9] = vertex_interpret(
                    grid_p[1].clone(),
                    grid_p[5].clone(),
                    min_vector.clone(),
                    baselen,
                    valuemap,
                    edgepos,
                    edgemap,
                ) as isize;
            }
            if EDGE_TABLE[cubeindex] & 1024 != 0 {
                vertlist[10] = vertex_interpret(
                    grid_p[2].clone(),
                    grid_p[6].clone(),
                    min_vector.clone(),
                    baselen,
                    valuemap,
                    edgepos,
                    edgemap,
                ) as isize;
            }
            if EDGE_TABLE[cubeindex] & 2048 != 0 {
                vertlist[11] = vertex_interpret(
                    grid_p[3].clone(),
                    grid_p[7].clone(),
                    min_vector.clone(),
                    baselen,
                    valuemap,
                    edgepos,
                    edgemap,
                ) as isize;
            }

            let mut i = 0;
            while TRI_TABLE[cubeindex][i] != -1 {
                let current = TRI_TABLE[cubeindex];
                matrix.push((vertlist[current[i] as usize], vertlist[current[i+1] as usize], vertlist[current[i+2] as usize]));
                i += 3;
            }
        }


        fs::create_dir_all("./render").unwrap();
        //Create output.obj if don't exist
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(format!("./render/water_{}", self.frame))
            .unwrap();

        if let Err(e) = writeln!(file, "Shape \"trianglemesh\"  \"integer indices\" [") {
            eprintln!("Couldn't write to file: {}", e);
        }

            /*let to_append = format!("{} {} {} ", input.0, input.1, input.2);
            data.push_str(&to_append);*/
        
        for input in matrix {
            if let Err(e) = writeln!(file, "{} {} {}", input.0, input.1, input.2) {
                eprintln!("Couldn't write to file: {}", e);
            }
        }
        if let Err(e) = writeln!(file, "]  \"point P\" [") {
            eprintln!("Couldn't write to file: {}", e);
        }
        
        for p in &self.edgepos {
            if let Err(e) = writeln!(file, "{} {} {} ", p.get_x(), p.get_z(), p.get_y()) {
                eprintln!("Couldn't write to file: {}", e);
            }
        }
        if let Err(e) = writeln!(file, "]") {
            eprintln!("Couldn't write to file: {}", e);
        }

        //clear
        self.edgepos = Vec::new();
        self.edgemap = HashMap::new();
        self.valuemap = HashMap::new();
        self.gridset = BTreeSet::new();

    }
}
