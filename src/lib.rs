#![allow(arithmetic_overflow)]

mod lookup;

fn fast_min(a: f32, b: f32) -> f32 {
    return if a < b { a } else { b };
}

fn fast_max(a: f32, b: f32) -> f32 {
    return if a > b { a } else { b };
}

fn fast_abs(f: f32) -> f32 {
    return if f < 0.0 { -f } else { f };
}

fn fast_sqrt(f: f32) -> f32 {
    return f.sqrt();
}

fn fast_floor(f: f32) -> i32 {
    return if f >= 0.0 { f as i32 } else { f as i32 - 1 };
}

fn fast_round(f: f32) -> i32 {
    return if f >= 0.0 {
        (f + 0.5) as i32
    } else {
        (f - 0.5) as i32
    };
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + t * (b - a);
}

fn interp_hermite(t: f32) -> f32 {
    return t * t * (3.0 - 2.0 * t);
}

fn interp_quintic(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn cubic_lerp(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let p = (d - c) - (a - b);
    return t * t * t * p + t * t * ((a - b) - p) + t * (c - a) + b;
}

fn ping_pong(mut t: f32) -> f32 {
    t -= ((t * 0.5) as i32 * 2) as f32;
    return if t < 1.0 { t } else { 2.0 - t };
}

pub enum NoiseType {
    OpenSimplex2,
    OpenSimplex2S,
    Cellular,
    Perlin,
    ValueCubic,
    Value,
}

pub enum RotationType3D {
    None,
    ImproveXYPlanes,
    ImproveXZPlanes,
}

pub enum FractalType {
    None,
    FBm,
    Ridged,
    PingPong,
    DomainWarpProgressive,
    DomainWarpIndependent,
}

#[derive(PartialEq)]
pub enum CellularDistanceFunction {
    Euclidean,
    EuclideanSq,
    Manhattan,
    Hybrid,
}

#[derive(PartialEq, PartialOrd)]
pub enum CellularReturnType {
    CellValue,
    Distance,
    Distance2,
    Distance2Add,
    Distance2Sub,
    Distance2Mul,
    Distance2Div,
}

pub enum DomainWarpType {
    OpenSimplex2,
    OpenSimplex2Reduced,
    BasicGrid,
}

pub enum TransformType3D {
    None,
    ImproveXYPlanes,
    ImproveXZPlanes,
    DefaultOpenSimplex2,
}

pub struct FastNoiseLite {
    seed: i32,
    frequency: f32,
    noise_type: NoiseType,
    rotation_type3d: RotationType3D,
    transform_type3d: TransformType3D,

    fractal_type: FractalType,
    octaves: usize,
    lacunarity: f32,
    gain: f32,
    weighted_strength: f32,
    ping_pong_strength: f32,

    fractal_bounding: f32,

    cellular_distance_function: CellularDistanceFunction,
    cellular_return_type: CellularReturnType,
    cellular_jitter_modifier: f32,

    domain_warp_type: DomainWarpType,
    warp_transform_type3d: TransformType3D,
    domain_warp_amp: f32,
}

impl FastNoiseLite {
    pub fn new(seed: i32) -> Self {
        Self {
            seed,
            frequency: 0.01,
            noise_type: NoiseType::OpenSimplex2,
            rotation_type3d: RotationType3D::None,
            transform_type3d: TransformType3D::DefaultOpenSimplex2,
            fractal_type: FractalType::None,
            octaves: 3,
            lacunarity: 2.0,
            gain: 0.5,
            weighted_strength: 0.0,
            ping_pong_strength: 2.0,
            fractal_bounding: 1.0 / 1.75,
            cellular_distance_function: CellularDistanceFunction::EuclideanSq,
            cellular_return_type: CellularReturnType::Distance,
            cellular_jitter_modifier: 1.0,
            domain_warp_type: DomainWarpType::OpenSimplex2,
            warp_transform_type3d: TransformType3D::DefaultOpenSimplex2,
            domain_warp_amp: 1.0,
        }
    }
    pub fn set_seed(&mut self, seed: i32) {
        self.seed = seed;
    }
    pub fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency;
    }
    pub fn set_noise_type(&mut self, noise_type: NoiseType) {
        self.noise_type = noise_type;
        self.update_transform_type_3d();
    }
    pub fn set_rotation_type_3d(&mut self, rotation_type3d: RotationType3D) {
        self.rotation_type3d = rotation_type3d;
        self.update_transform_type_3d();
        self.update_warp_transform_type_3d();
    }
    pub fn set_fractal_type(&mut self, fractal_type: FractalType) {
        self.fractal_type = fractal_type;
    }
    pub fn set_fractal_octaves(&mut self, octaves: usize) {
        self.octaves = octaves;
        self.calculate_fractal_bounding();
    }
    pub fn set_fractal_lacunarity(&mut self, lacunarity: f32) {
        self.lacunarity = lacunarity;
    }
    pub fn set_fractal_gain(&mut self, gain: f32) {
        self.gain = gain;
        self.calculate_fractal_bounding();
    }

    pub fn set_fractal_weighted_strength(&mut self, weighted_strength: f32) {
        self.weighted_strength = weighted_strength;
    }

    pub fn set_fractal_ping_pong_strength(&mut self, ping_pong_strength: f32) {
        self.ping_pong_strength = ping_pong_strength;
    }

    pub fn set_cellular_distance_function(
        &mut self,
        cellular_distance_function: CellularDistanceFunction,
    ) {
        self.cellular_distance_function = cellular_distance_function;
    }

    pub fn set_cellular_return_type(&mut self, cellular_return_type: CellularReturnType) {
        self.cellular_return_type = cellular_return_type;
    }

    pub fn set_cellular_jitter(&mut self, cellular_jitter: f32) {
        self.cellular_jitter_modifier = cellular_jitter;
    }

    pub fn set_domain_warp_type(&mut self, domain_warp_type: DomainWarpType) {
        self.domain_warp_type = domain_warp_type;
        self.update_warp_transform_type_3d();
    }

    pub fn set_domain_warp_amp(&mut self, domain_warp_amp: f32) {
        self.domain_warp_amp = domain_warp_amp;
    }

    pub fn get_noise_2d(&self, mut x: f32, mut y: f32) -> f32 {
        self.transform_noise_coordinate_2d(&mut x, &mut y);

        match self.fractal_type {
            FractalType::FBm => self.gen_fractal_fbm_2d(x, y),
            FractalType::Ridged => self.gen_fractal_ridged_2d(x, y),
            FractalType::PingPong => self.gen_fractal_ping_pong_2d(x, y),
            _ => self.gen_noise_single_2d(self.seed, x, y),
        }
    }

    pub fn get_noise_3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        self.transform_noise_coordinate_3d(&mut x, &mut y, &mut z);
        match self.fractal_type {
            FractalType::FBm => self.gen_fractal_fbm_3d(x, y, z),
            FractalType::Ridged => self.gen_fractal_ridged_3d(x, y, z),
            FractalType::PingPong => self.gen_fractal_ping_pong_3d(x, y, z),
            _ => self.gen_noise_single_3d(self.seed, x, y, z),
        }
    }

    pub fn domain_warp_2d(&self, x: &mut f32, y: &mut f32) {
        match self.fractal_type {
            FractalType::DomainWarpProgressive => self.domain_warp_fractal_progressive_2d(x, y),
            FractalType::DomainWarpIndependent => self.domain_warp_fractal_independent_2d(x, y),
            _ => self.domain_warp_single_2d(x, y),
        }
    }
    pub fn domain_warp_3d(&self, x: &mut f32, y: &mut f32, z: &mut f32) {
        match self.fractal_type {
            FractalType::DomainWarpIndependent => self.domain_warp_fractal_independent_3d(x, y, z),
            FractalType::DomainWarpProgressive => self.domain_warp_fractal_progressive_3d(x, y, z),
            _ => self.domain_warp_single_3d(x, y, z),
        }
    }
    fn calculate_fractal_bounding(&mut self) {
        let gain = fast_abs(self.gain);
        let mut amp = gain;
        let mut amp_fractal = 1.0;
        for _ in 0..self.octaves {
            amp_fractal += amp;
            amp *= gain;
        }
        self.fractal_bounding = 1.0 / amp_fractal;
    }
    const PRIME_X: i32 = 501125321;
    const PRIME_Y: i32 = 1136930381;
    const PRIME_Z: i32 = 1720413743;

    fn hash_2d(seed: i32, x_primed: i32, y_primed: i32) -> i32 {
        let mut hash = seed ^ x_primed ^ y_primed;
        hash = hash.wrapping_mul(0x27d4eb2d);
        return hash;
    }

    fn hash_3d(seed: i32, x_primed: i32, y_primed: i32, z_primed: i32) -> i32 {
        let mut hash = seed ^ x_primed ^ y_primed ^ z_primed;
        hash = hash.wrapping_mul(0x27d4eb2d);
        return hash;
    }

    fn val_coord_2d(seed: i32, x_primed: i32, y_primed: i32) -> f32 {
        let mut hash: i32 = Self::hash_2d(seed, x_primed, y_primed);

        hash *= hash;
        hash ^= hash << 19;
        return hash as f32 * (1.0 / 2147483648.0);
    }

    fn val_coord_3d(seed: i32, x_primed: i32, y_primed: i32, z_primed: i32) -> f32 {
        let mut hash: i32 = Self::hash_3d(seed, x_primed, y_primed, z_primed);

        hash *= hash;
        hash ^= hash << 19;
        return hash as f32 * (1.0 / 2147483648.0);
    }

    fn grad_coord_2d(seed: i32, x_primed: i32, y_primed: i32, xd: f32, yd: f32) -> f32 {
        let mut hash = Self::hash_2d(seed, x_primed, y_primed);
        hash ^= hash >> 15;
        hash &= 127 << 1;

        let xg = lookup::GRADIENTS_2D[hash as usize];
        let yg = lookup::GRADIENTS_2D[(hash | 1) as usize];

        return xd * xg + yd * yg;
    }

    fn grad_coord_3d(
        seed: i32,
        x_primed: i32,
        y_primed: i32,
        z_primed: i32,
        xd: f32,
        yd: f32,
        zd: f32,
    ) -> f32 {
        let mut hash: i32 = Self::hash_3d(seed, x_primed, y_primed, z_primed);
        hash ^= hash >> 15;
        hash &= 63 << 2;

        let xg: f32 = lookup::GRADIENTS_3D[hash as usize];
        let yg: f32 = lookup::GRADIENTS_3D[(hash | 1) as usize];
        let zg: f32 = lookup::GRADIENTS_3D[(hash | 2) as usize];

        return xd * xg + yd * yg + zd * zg;
    }

    fn grad_coord_out_2d(seed: i32, x_primed: i32, y_primed: i32, xo: &mut f32, yo: &mut f32) {
        let hash: i32 = Self::hash_2d(seed, x_primed, y_primed) & (255 << 1);

        *xo = lookup::RAND_VECS_2D[hash as usize];
        *yo = lookup::RAND_VECS_2D[(hash | 1) as usize];
    }

    fn grad_coord_out_3d(
        seed: i32,
        x_primed: i32,
        y_primed: i32,
        z_primed: i32,
        xo: &mut f32,
        yo: &mut f32,
        zo: &mut f32,
    ) {
        let hash: i32 = Self::hash_3d(seed, x_primed, y_primed, z_primed) & (255 << 2);

        *xo = lookup::RAND_VECS_3D[hash as usize];
        *yo = lookup::RAND_VECS_3D[(hash | 1) as usize];
        *zo = lookup::RAND_VECS_3D[(hash | 2) as usize];
    }

    fn grad_coord_dual_2d(
        seed: i32,
        x_primed: i32,
        y_primed: i32,
        xd: f32,
        yd: f32,
        xo: &mut f32,
        yo: &mut f32,
    ) {
        let hash: i32 = Self::hash_2d(seed, x_primed, y_primed);
        let index1: i32 = hash & (127 << 1);
        let index2: i32 = (hash >> 7) & (255 << 1);

        let xg: f32 = lookup::GRADIENTS_2D[index1 as usize];
        let yg: f32 = lookup::GRADIENTS_2D[(index1 | 1) as usize];
        let value: f32 = xd * xg + yd * yg;

        let xgo: f32 = lookup::RAND_VECS_2D[index2 as usize];
        let ygo: f32 = lookup::RAND_VECS_2D[(index2 | 1) as usize];

        *xo = value * xgo;
        *yo = value * ygo;
    }

    fn grad_coord_dual_3d(
        seed: i32,
        x_primed: i32,
        y_primed: i32,
        z_primed: i32,
        xd: f32,
        yd: f32,
        zd: f32,
        xo: &mut f32,
        yo: &mut f32,
        zo: &mut f32,
    ) {
        let hash: i32 = Self::hash_3d(seed, x_primed, y_primed, z_primed);
        let index1: i32 = hash & (63 << 2);
        let index2: i32 = (hash >> 6) & (255 << 2);

        let xg: f32 = lookup::GRADIENTS_3D[index1 as usize];
        let yg: f32 = lookup::GRADIENTS_3D[(index1 | 1) as usize];
        let zg: f32 = lookup::GRADIENTS_3D[(index1 | 2) as usize];
        let value: f32 = xd * xg + yd * yg + zd * zg;

        let xgo: f32 = lookup::RAND_VECS_3D[index2 as usize];
        let ygo: f32 = lookup::RAND_VECS_3D[(index2 | 1) as usize];
        let zgo: f32 = lookup::RAND_VECS_3D[(index2 | 2) as usize];

        *xo = value * xgo;
        *yo = value * ygo;
        *zo = value * zgo;
    }
    fn gen_noise_single_2d(&self, seed: i32, x: f32, y: f32) -> f32 {
        match self.noise_type {
            NoiseType::OpenSimplex2 => self.single_open_simplex_2_2d(seed, x, y),
            NoiseType::OpenSimplex2S => self.single_open_simplex2s_2d(seed, x, y),
            NoiseType::Cellular => self.single_cellular_2d(seed, x, y),
            NoiseType::Perlin => self.single_perlin_2d(seed, x, y),
            NoiseType::ValueCubic => self.single_value_cubic_2d(seed, x, y),
            NoiseType::Value => self.single_value_2d(seed, x, y),
        }
    }

    fn gen_noise_single_3d(&self, seed: i32, x: f32, y: f32, z: f32) -> f32 {
        match self.noise_type {
            NoiseType::OpenSimplex2 => self.single_open_simplex_2_3d(seed, x, y, z),
            NoiseType::OpenSimplex2S => self.single_open_simplex_2s_3d(seed, x, y, z),
            NoiseType::Cellular => self.single_cellular_3d(seed, x, y, z),
            NoiseType::Perlin => self.single_perlin_3d(seed, x, y, z),
            NoiseType::ValueCubic => self.single_value_cubic_3d(seed, x, y, z),
            NoiseType::Value => self.single_value_3d(seed, x, y, z),
        }
    }

    fn transform_noise_coordinate_2d(&self, x: &mut f32, y: &mut f32) {
        *x *= self.frequency;
        *y *= self.frequency;

        match self.noise_type {
            NoiseType::OpenSimplex2 | NoiseType::OpenSimplex2S => {
                const SQRT3: f32 = 1.7320508075688772935274463415059;
                const F2: f32 = 0.5 * (SQRT3 - 1.0);
                let t: f32 = (*x + *y) * F2;
                *x += t;
                *y += t;
            }
            _ => {}
        }
    }

    fn transform_noise_coordinate_3d(&self, x: &mut f32, y: &mut f32, z: &mut f32) {
        *x *= self.frequency;
        *y *= self.frequency;
        *z *= self.frequency;

        match self.transform_type3d {
            TransformType3D::ImproveXYPlanes => {
                let xy: f32 = *x + *y;
                let s2: f32 = xy * -0.211324865405187;
                *z *= 0.577350269189626;
                *x += s2 - *z;
                *y = *y + s2 - *z;
                *z += xy * 0.577350269189626;
            }
            TransformType3D::ImproveXZPlanes => {
                let xz: f32 = *x + *z;
                let s2: f32 = xz * -0.211324865405187;
                *y *= 0.577350269189626;
                *x += s2 - *y;
                *z += s2 - *y;
                *y += xz * 0.577350269189626;
            }
            TransformType3D::DefaultOpenSimplex2 => {
                const R3: f32 = 2.0 / 3.0;
                let r: f32 = (*x + *y + *z) * R3; // Rotation, not skew
                *x = r - *x;
                *y = r - *y;
                *z = r - *z;
            }
            TransformType3D::None => {}
        }
    }

    fn update_transform_type_3d(&mut self) {
        match self.rotation_type3d {
            RotationType3D::ImproveXYPlanes => {
                self.transform_type3d = TransformType3D::ImproveXYPlanes;
            }
            RotationType3D::ImproveXZPlanes => {
                self.transform_type3d = TransformType3D::ImproveXZPlanes;
            }
            RotationType3D::None => match self.noise_type {
                NoiseType::OpenSimplex2 | NoiseType::OpenSimplex2S => {
                    self.transform_type3d = TransformType3D::DefaultOpenSimplex2;
                }
                _ => {
                    self.transform_type3d = TransformType3D::None;
                }
            },
        };
    }

    // Domain Warp Coordinate Transforms

    fn transform_domain_warp_coordinate_2d(&self, x: &mut f32, y: &mut f32) {
        match self.domain_warp_type {
            DomainWarpType::OpenSimplex2 | DomainWarpType::OpenSimplex2Reduced => {
                const SQRT3: f32 = 1.7320508075688772935274463415059;
                const F2: f32 = 0.5 * (SQRT3 - 1.0);
                let t: f32 = (*x + *y) * F2;
                *x += t;
                *y += t;
            }
            _ => {}
        }
    }

    fn transform_domain_warp_coordinate_3d(&self, x: &mut f32, y: &mut f32, z: &mut f32) {
        match self.warp_transform_type3d {
            TransformType3D::ImproveXYPlanes => {
                let xy: f32 = *x + *y;
                let s2: f32 = xy * -0.211324865405187;
                *z *= 0.577350269189626;
                *x += s2 - *z;
                *y = *y + s2 - *z;
                *z += xy * 0.577350269189626;
            }
            TransformType3D::ImproveXZPlanes => {
                let xz: f32 = *x + *z;
                let s2: f32 = xz * -0.211324865405187;
                *y *= 0.577350269189626;
                *x += s2 - *y;
                *z += s2 - *y;
                *y += xz * 0.577350269189626;
            }
            TransformType3D::DefaultOpenSimplex2 => {
                const R3: f32 = 2.0 / 3.0;
                let r: f32 = (*x + *y + *z) * R3; // Rotation, not skew
                *x = r - *x;
                *y = r - *y;
                *z = r - *z;
            }
            _ => {}
        }
    }

    fn update_warp_transform_type_3d(&mut self) {
        match self.rotation_type3d {
            RotationType3D::ImproveXYPlanes => {
                self.warp_transform_type3d = TransformType3D::ImproveXYPlanes
            }
            RotationType3D::ImproveXZPlanes => {
                self.warp_transform_type3d = TransformType3D::ImproveXZPlanes
            }
            RotationType3D::None => match self.domain_warp_type {
                DomainWarpType::OpenSimplex2 | DomainWarpType::OpenSimplex2Reduced => {
                    self.warp_transform_type3d = TransformType3D::DefaultOpenSimplex2
                }
                _ => self.warp_transform_type3d = TransformType3D::None,
            },
        }
    }

    fn gen_fractal_fbm_2d(&self, mut x: f32, mut y: f32) -> f32 {
        let mut seed = self.seed;
        let mut sum = 0.0;
        let mut amp = self.fractal_bounding;

        for _ in 0..self.octaves {
            let noise = self.gen_noise_single_2d(seed, x, y);
            seed += 1;
            sum += noise * amp;
            amp *= lerp(
                1.0,
                fast_min(noise + 1.0, 2.0) * 0.5,
                self.weighted_strength,
            );

            x *= self.lacunarity;
            y *= self.lacunarity;
            amp *= self.gain;
        }

        return sum;
    }

    fn gen_fractal_fbm_3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        let mut seed = self.seed;
        let mut sum = 0.0;
        let mut amp = self.fractal_bounding;

        for _ in 0..self.octaves {
            let noise = self.gen_noise_single_3d(seed, x, y, z);
            seed += 1;
            sum += noise * amp;
            amp *= lerp(1.0, (noise + 1.0) * 0.5, self.weighted_strength);

            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;
            amp *= self.gain;
        }

        return sum;
    }

    fn gen_fractal_ridged_2d(&self, mut x: f32, mut y: f32) -> f32 {
        let mut seed = self.seed;
        let mut sum = 0.0;
        let mut amp = self.fractal_bounding;

        for _ in 0..self.octaves {
            let noise = fast_abs(self.gen_noise_single_2d(seed, x, y));
            seed += 1;
            sum += (noise * -2.0 + 1.0) * amp;
            amp *= lerp(1.0, 1.0 - noise, self.weighted_strength);

            x *= self.lacunarity;
            y *= self.lacunarity;
            amp *= self.gain;
        }

        return sum;
    }

    fn gen_fractal_ridged_3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        let mut seed = self.seed;
        let mut sum = 0.0;
        let mut amp = self.fractal_bounding;

        for _ in 0..self.octaves {
            let noise = fast_abs(self.gen_noise_single_3d(seed, x, y, z));
            seed += 1;
            sum += (noise * -2.0 + 1.0) * amp;
            amp *= lerp(1.0, 1.0 - noise, self.weighted_strength);

            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;
            amp *= self.gain;
        }

        return sum;
    }

    fn gen_fractal_ping_pong_2d(&self, mut x: f32, mut y: f32) -> f32 {
        let mut seed = self.seed;
        let mut sum = 0.0;
        let mut amp = self.fractal_bounding;

        for _ in 0..self.octaves {
            let noise =
                ping_pong((self.gen_noise_single_2d(seed, x, y) + 1.0) * self.ping_pong_strength);
            seed += 1;
            sum += (noise - 0.5) * 2.0 * amp;
            amp *= lerp(1.0, noise, self.weighted_strength);

            x *= self.lacunarity;
            y *= self.lacunarity;
            amp *= self.gain;
        }

        return sum;
    }

    fn gen_fractal_ping_pong_3d(&self, mut x: f32, mut y: f32, mut z: f32) -> f32 {
        let mut seed = self.seed;
        let mut sum = 0.0;
        let mut amp = self.fractal_bounding;

        for _ in 0..self.octaves {
            let noise = ping_pong(
                (self.gen_noise_single_3d(seed, x, y, z) + 1.0) * self.ping_pong_strength,
            );
            seed += 1;
            sum += (noise - 0.5) * 2.0 * amp;
            amp *= lerp(1.0, noise, self.weighted_strength);

            x *= self.lacunarity;
            y *= self.lacunarity;
            z *= self.lacunarity;
            amp *= self.gain;
        }

        return sum;
    }
    fn single_open_simplex_2_2d(&self, seed: i32, x: f32, y: f32) -> f32 {
        // 2D OpenSimplex2 case uses the same algorithm as ordinary Simplex.

        const SQRT3: f32 = 1.7320508075688772935274463415059;
        const G2: f32 = (3.0 - SQRT3) / 6.0;

        /*
         * --- Skew moved to TransformNoiseCoordinate method ---
         * const F2: f32 = 0.5f * (SQRT3 - 1);
         * s: f32 = (x + y) * F2;
         * x += s; y += s;
         */

        let mut i: i32 = fast_floor(x);
        let mut j: i32 = fast_floor(y);
        let xi: f32 = x - i as f32;
        let yi: f32 = y - j as f32;

        let t: f32 = (xi + yi) * G2;
        let x0: f32 = xi - t;
        let y0: f32 = yi - t;

        i = i.wrapping_mul(Self::PRIME_X);
        j = j.wrapping_mul(Self::PRIME_Y);

        let n0: f32;
        let n1: f32;
        let n2: f32;

        let a: f32 = 0.5 - x0 * x0 - y0 * y0;
        if a <= 0.0 {
            n0 = 0.0;
        } else {
            n0 = (a * a) * (a * a) * Self::grad_coord_2d(seed, i, j, x0, y0);
        }

        let c: f32 = (2.0 * (1.0 - 2.0 * G2) * (1.0 / G2 - 2.0)) * t
            + ((-2.0 * (1.0 - 2.0 * G2) * (1.0 - 2.0 * G2)) + a);
        if c <= 0.0 {
            n2 = 0.0;
        } else {
            let x2: f32 = x0 + (2.0 * G2 - 1.0);
            let y2: f32 = y0 + (2.0 * G2 - 1.0);
            n2 = (c * c)
                * (c * c)
                * Self::grad_coord_2d(
                    seed,
                    i.wrapping_add(Self::PRIME_X),
                    j.wrapping_add(Self::PRIME_Y),
                    x2,
                    y2,
                );
        }

        if y0 > x0 {
            let x1: f32 = x0 + G2;
            let y1: f32 = y0 + (G2 - 1.0);
            let b: f32 = 0.5 - x1 * x1 - y1 * y1;
            if b <= 0.0 {
                n1 = 0.0;
            } else {
                n1 = (b * b)
                    * (b * b)
                    * Self::grad_coord_2d(seed, i, j.wrapping_add(Self::PRIME_Y), x1, y1);
            }
        } else {
            let x1: f32 = x0 + (G2 - 1.0);
            let y1: f32 = y0 + G2;
            let b: f32 = 0.5 - x1 * x1 - y1 * y1;
            if b <= 0.0 {
                n1 = 0.0;
            } else {
                n1 = (b * b)
                    * (b * b)
                    * Self::grad_coord_2d(seed, i.wrapping_add(Self::PRIME_X), j, x1, y1);
            }
        }

        return (n0 + n1 + n2) * 99.83685446303647;
    }

    fn single_open_simplex_2_3d(&self, mut seed: i32, x: f32, y: f32, z: f32) -> f32 {
        // 3D OpenSimplex2 case uses two offset rotated cube grids.

        /*
         * --- Rotation moved to TransformNoiseCoordinate method ---
         * const R3: f32 = (FNfloat)(2.0 / 3.0);
         * r: f32 = (x + y + z) * R3; // Rotation, not skew
         * x = r - x; y = r - y; z = r - z;
         */

        let mut i: i32 = fast_round(x);
        let mut j: i32 = fast_round(y);
        let mut k: i32 = fast_round(z);
        let mut x0: f32 = x - i as f32;
        let mut y0: f32 = y - j as f32;
        let mut z0: f32 = z - k as f32;

        let mut x_nsign: i32 = (-1.0 - x0) as i32 | 1;
        let mut y_nsign: i32 = (-1.0 - y0) as i32 | 1;
        let mut z_nsign: i32 = (-1.0 - z0) as i32 | 1;

        let mut ax0: f32 = x_nsign as f32 * -x0;
        let mut ay0: f32 = y_nsign as f32 * -y0;
        let mut az0: f32 = z_nsign as f32 * -z0;

        i = i.wrapping_mul(Self::PRIME_X);
        j = j.wrapping_mul(Self::PRIME_Y);
        k = k.wrapping_mul(Self::PRIME_Z);

        let mut value: f32 = 0.0;
        let mut a: f32 = (0.6 - x0 * x0) - (y0 * y0 + z0 * z0);

        for l in 0..2 {
            if a > 0.0 {
                value += (a * a) * (a * a) * Self::grad_coord_3d(seed, i, j, k, x0, y0, z0);
            }

            let mut b: f32 = a + 1.0;
            let mut i1: i32 = i;
            let mut j1: i32 = j;
            let mut k1: i32 = k;
            let mut x1: f32 = x0;
            let mut y1: f32 = y0;
            let mut z1: f32 = z0;

            if ax0 >= ay0 && ax0 >= az0 {
                x1 += x_nsign as f32;
                b -= x_nsign as f32 * 2.0 * x1;
                i1 -= x_nsign * Self::PRIME_X;
            } else if ay0 > ax0 && ay0 >= az0 {
                y1 += y_nsign as f32;
                b -= y_nsign as f32 * 2.0 * y1;
                j1 -= y_nsign * Self::PRIME_Y;
            } else {
                z1 += z_nsign as f32;
                b -= z_nsign as f32 * 2.0 * z1;
                k1 -= z_nsign * Self::PRIME_Z;
            }

            if b > 0.0 {
                value += (b * b) * (b * b) * Self::grad_coord_3d(seed, i1, j1, k1, x1, y1, z1);
            }

            if l == 1 {
                break;
            }

            ax0 = 0.5 - ax0;
            ay0 = 0.5 - ay0;
            az0 = 0.5 - az0;

            x0 = x_nsign as f32 * ax0;
            y0 = y_nsign as f32 * ay0;
            z0 = z_nsign as f32 * az0;

            a += (0.75 - ax0) - (ay0 + az0);

            i += (x_nsign >> 1) & Self::PRIME_X;
            j += (y_nsign >> 1) & Self::PRIME_Y;
            k += (z_nsign >> 1) & Self::PRIME_Z;

            x_nsign = -x_nsign;
            y_nsign = -y_nsign;
            z_nsign = -z_nsign;

            seed = !seed;
        }

        return value * 32.69428253173828125;
    }

    // OpenSimplex2S Noise

    fn single_open_simplex2s_2d(&self, seed: i32, x: f32, y: f32) -> f32 {
        // 2D OpenSimplex2S case is a modified 2D simplex noise.

        const SQRT3: f32 = 1.7320508075688772935274463415059;
        const G2: f32 = (3.0 - SQRT3) / 6.0;

        /*
         * --- Skew moved to TransformNoiseCoordinate method ---
         * const F2: f32 = 0.5f * (SQRT3 - 1);
         * s: f32 = (x + y) * F2;
         * x += s; y += s;
         */

        let mut i: i32 = fast_floor(x);
        let mut j: i32 = fast_floor(y);
        let xi: f32 = x - i as f32;
        let yi: f32 = y - j as f32;

        i = i.wrapping_mul(Self::PRIME_X);
        j = j.wrapping_mul(Self::PRIME_Y);
        let i1: i32 = i.wrapping_add(Self::PRIME_X);
        let j1: i32 = j.wrapping_add(Self::PRIME_Y);

        let t: f32 = (xi + yi) * G2;
        let x0: f32 = xi - t;
        let y0: f32 = yi - t;

        let a0: f32 = (2.0 / 3.0) - x0 * x0 - y0 * y0;
        let mut value: f32 = (a0 * a0) * (a0 * a0) * Self::grad_coord_2d(seed, i, j, x0, y0);

        let a1: f32 = (2.0 * (1.0 - 2.0 * G2) * (1.0 / G2 - 2.0)) * t
            + ((-2.0 * (1.0 - 2.0 * G2) * (1.0 - 2.0 * G2)) + a0);
        let x1: f32 = x0 - (1.0 - 2.0 * G2);
        let y1: f32 = y0 - (1.0 - 2.0 * G2);
        value += (a1 * a1) * (a1 * a1) * Self::grad_coord_2d(seed, i1, j1, x1, y1);

        // Nested conditionals were faster than compact bit logic/arithmetic.
        let xmyi: f32 = xi - yi;
        if t > G2 {
            if xi + xmyi > 1.0 {
                let x2: f32 = x0 + (3.0 * G2 - 2.0);
                let y2: f32 = y0 + (3.0 * G2 - 1.0);
                let a2: f32 = (2.0 / 3.0) - x2 * x2 - y2 * y2;
                if a2 > 0.0 {
                    value += (a2 * a2)
                        * (a2 * a2)
                        * Self::grad_coord_2d(
                            seed,
                            i.wrapping_add(Self::PRIME_X << 1),
                            j.wrapping_add(Self::PRIME_Y),
                            x2,
                            y2,
                        );
                }
            } else {
                let x2: f32 = x0 + G2;
                let y2: f32 = y0 + (G2 - 1.0);
                let a2: f32 = (2.0 / 3.0) - x2 * x2 - y2 * y2;
                if a2 > 0.0 {
                    value += (a2 * a2)
                        * (a2 * a2)
                        * Self::grad_coord_2d(seed, i, j.wrapping_add(Self::PRIME_Y), x2, y2);
                }
            }

            if yi - xmyi > 1.0 {
                let x3: f32 = x0 + (3.0 * G2 - 1.0);
                let y3: f32 = y0 + (3.0 * G2 - 2.0);
                let a3: f32 = (2.0 / 3.0) - x3 * x3 - y3 * y3;
                if a3 > 0.0 {
                    value += (a3 * a3)
                        * (a3 * a3)
                        * Self::grad_coord_2d(
                            seed,
                            i.wrapping_add(Self::PRIME_X),
                            j.wrapping_add(Self::PRIME_Y << 1),
                            x3,
                            y3,
                        );
                }
            } else {
                let x3: f32 = x0 + (G2 - 1.0);
                let y3: f32 = y0 + G2;
                let a3: f32 = (2.0 / 3.0) - x3 * x3 - y3 * y3;
                if a3 > 0.0 {
                    value += (a3 * a3)
                        * (a3 * a3)
                        * Self::grad_coord_2d(seed, i.wrapping_add(Self::PRIME_X), j, x3, y3);
                }
            }
        } else {
            if xi + xmyi < 0.0 {
                let x2: f32 = x0 + (1.0 - G2);
                let y2: f32 = y0 - G2;
                let a2: f32 = (2.0 / 3.0) - x2 * x2 - y2 * y2;
                if a2 > 0.0 {
                    value += (a2 * a2)
                        * (a2 * a2)
                        * Self::grad_coord_2d(seed, i.wrapping_sub(Self::PRIME_X), j, x2, y2);
                }
            } else {
                let x2: f32 = x0 + (G2 - 1.0);
                let y2: f32 = y0 + G2;
                let a2: f32 = (2.0 / 3.0) - x2 * x2 - y2 * y2;
                if a2 > 0.0 {
                    value += (a2 * a2)
                        * (a2 * a2)
                        * Self::grad_coord_2d(seed, i.wrapping_add(Self::PRIME_X), j, x2, y2);
                }
            }

            if yi < xmyi {
                let x2: f32 = x0 - G2;
                let y2: f32 = y0 - (G2 - 1.0);
                let a2: f32 = (2.0 / 3.0) - x2 * x2 - y2 * y2;
                if a2 > 0.0 {
                    value += (a2 * a2)
                        * (a2 * a2)
                        * Self::grad_coord_2d(seed, i, j.wrapping_sub(Self::PRIME_Y), x2, y2);
                }
            } else {
                let x2: f32 = x0 + G2;
                let y2: f32 = y0 + (G2 - 1.0);
                let a2: f32 = (2.0 / 3.0) - x2 * x2 - y2 * y2;
                if a2 > 0.0 {
                    value += (a2 * a2)
                        * (a2 * a2)
                        * Self::grad_coord_2d(seed, i, j.wrapping_add(Self::PRIME_Y), x2, y2);
                }
            }
        }

        return value * 18.24196194486065;
    }

    fn single_open_simplex_2s_3d(&self, seed: i32, x: f32, y: f32, z: f32) -> f32 {
        // 3D OpenSimplex2S case uses two offset rotated cube grids.

        /*
         * --- Rotation moved to TransformNoiseCoordinate method ---
         * const R3: f32 = (FNfloat)(2.0 / 3.0);
         * r: f32 = (x + y + z) * R3; // Rotation, not skew
         * x = r - x; y = r - y; z = r - z;
         */

        let mut i: i32 = fast_floor(x);
        let mut j: i32 = fast_floor(y);
        let mut k: i32 = fast_floor(z);
        let xi: f32 = x - i as f32;
        let yi: f32 = y - j as f32;
        let zi: f32 = z - k as f32;

        i = i.wrapping_mul(Self::PRIME_X);
        j = j.wrapping_mul(Self::PRIME_Y);
        k = k.wrapping_mul(Self::PRIME_Z);
        let seed2: i32 = seed.wrapping_add(1293373);

        let x_n_mask: i32 = (-0.5 - xi) as i32;
        let y_n_mask: i32 = (-0.5 - yi) as i32;
        let z_n_mask: i32 = (-0.5 - zi) as i32;

        let x0: f32 = xi + x_n_mask as f32;
        let y0: f32 = yi + y_n_mask as f32;
        let z0: f32 = zi + z_n_mask as f32;
        let a0: f32 = 0.75 - x0 * x0 - y0 * y0 - z0 * z0;
        let mut value: f32 = (a0 * a0)
            * (a0 * a0)
            * Self::grad_coord_3d(
                seed,
                i + (x_n_mask & Self::PRIME_X),
                j + (y_n_mask & Self::PRIME_Y),
                k + (z_n_mask & Self::PRIME_Z),
                x0,
                y0,
                z0,
            );

        let x1: f32 = xi - 0.5;
        let y1: f32 = yi - 0.5;
        let z1: f32 = zi - 0.5;
        let a1: f32 = 0.75 - x1 * x1 - y1 * y1 - z1 * z1;
        value += (a1 * a1)
            * (a1 * a1)
            * Self::grad_coord_3d(
                seed2,
                i.wrapping_add(Self::PRIME_X),
                j.wrapping_add(Self::PRIME_Y),
                k.wrapping_add(Self::PRIME_Z),
                x1,
                y1,
                z1,
            );

        let x_a_flip_mask_0: f32 = ((x_n_mask | 1) << 1) as f32 * x1;
        let y_a_flip_mask_0: f32 = ((y_n_mask | 1) << 1) as f32 * y1;
        let z_a_flip_mask_0: f32 = ((z_n_mask | 1) << 1) as f32 * z1;
        let x_a_flip_mask_1: f32 = (-2 - (x_n_mask << 2)) as f32 * x1 - 1.0;
        let y_a_flip_mask_1: f32 = (-2 - (y_n_mask << 2)) as f32 * y1 - 1.0;
        let z_a_flip_mask_1: f32 = (-2 - (z_n_mask << 2)) as f32 * z1 - 1.0;

        let mut skip_5 = false;
        let a2: f32 = x_a_flip_mask_0 + a0;
        if a2 > 0.0 {
            let x2: f32 = x0 - (x_n_mask | 1) as f32;
            let y2: f32 = y0;
            let z2: f32 = z0;
            value += (a2 * a2)
                * (a2 * a2)
                * Self::grad_coord_3d(
                    seed,
                    i + (!x_n_mask & Self::PRIME_X),
                    j + (y_n_mask & Self::PRIME_Y),
                    k + (z_n_mask & Self::PRIME_Z),
                    x2,
                    y2,
                    z2,
                );
        } else {
            let a3: f32 = y_a_flip_mask_0 + z_a_flip_mask_0 + a0;
            if a3 > 0.0 {
                let x3: f32 = x0;
                let y3: f32 = y0 - (y_n_mask | 1) as f32;
                let z3: f32 = z0 - (z_n_mask | 1) as f32;
                value += (a3 * a3)
                    * (a3 * a3)
                    * Self::grad_coord_3d(
                        seed,
                        i + (x_n_mask & Self::PRIME_X),
                        j + (!y_n_mask & Self::PRIME_Y),
                        k + (!z_n_mask & Self::PRIME_Z),
                        x3,
                        y3,
                        z3,
                    );
            }

            let a4: f32 = x_a_flip_mask_1 + a1;
            if a4 > 0.0 {
                let x4: f32 = (x_n_mask | 1) as f32 + x1;
                let y4: f32 = y1;
                let z4: f32 = z1;
                value += (a4 * a4)
                    * (a4 * a4)
                    * Self::grad_coord_3d(
                        seed2,
                        i + (x_n_mask & (Self::PRIME_X * 2)),
                        j.wrapping_add(Self::PRIME_Y),
                        k.wrapping_add(Self::PRIME_Z),
                        x4,
                        y4,
                        z4,
                    );
                skip_5 = true;
            }
        }

        let mut skip_9 = false;
        let a6: f32 = y_a_flip_mask_0 + a0;
        if a6 > 0.0 {
            let x6: f32 = x0;
            let y6: f32 = y0 - (y_n_mask | 1) as f32;
            let z6: f32 = z0;
            value += (a6 * a6)
                * (a6 * a6)
                * Self::grad_coord_3d(
                    seed,
                    i + (x_n_mask & Self::PRIME_X),
                    j + (!y_n_mask & Self::PRIME_Y),
                    k + (z_n_mask & Self::PRIME_Z),
                    x6,
                    y6,
                    z6,
                );
        } else {
            let a7: f32 = x_a_flip_mask_0 + z_a_flip_mask_0 + a0;
            if a7 > 0.0 {
                let x7: f32 = x0 - (x_n_mask | 1) as f32;
                let y7: f32 = y0;
                let z7: f32 = z0 - (z_n_mask | 1) as f32;
                value += (a7 * a7)
                    * (a7 * a7)
                    * Self::grad_coord_3d(
                        seed,
                        i + (!x_n_mask & Self::PRIME_X),
                        j + (y_n_mask & Self::PRIME_Y),
                        k + (!z_n_mask & Self::PRIME_Z),
                        x7,
                        y7,
                        z7,
                    );
            }

            let a8: f32 = y_a_flip_mask_1 + a1;
            if a8 > 0.0 {
                let x8: f32 = x1;
                let y8: f32 = (y_n_mask | 1) as f32 + y1;
                let z8: f32 = z1;
                value += (a8 * a8)
                    * (a8 * a8)
                    * Self::grad_coord_3d(
                        seed2,
                        i.wrapping_add(Self::PRIME_X),
                        j + (y_n_mask & (Self::PRIME_Y << 1)),
                        k.wrapping_add(Self::PRIME_Z),
                        x8,
                        y8,
                        z8,
                    );
                skip_9 = true;
            }
        }

        let mut skip_d = false;
        let a_a: f32 = z_a_flip_mask_0 + a0;
        if a_a > 0.0 {
            let x_a: f32 = x0;
            let y_a: f32 = y0;
            let z_a: f32 = z0 - (z_n_mask | 1) as f32;
            value += (a_a * a_a)
                * (a_a * a_a)
                * Self::grad_coord_3d(
                    seed,
                    i + (x_n_mask & Self::PRIME_X),
                    j + (y_n_mask & Self::PRIME_Y),
                    k + (!z_n_mask & Self::PRIME_Z),
                    x_a,
                    y_a,
                    z_a,
                );
        } else {
            let a_b: f32 = x_a_flip_mask_0 + y_a_flip_mask_0 + a0;
            if a_b > 0.0 {
                let x_b: f32 = x0 - (x_n_mask | 1) as f32;
                let y_b: f32 = y0 - (y_n_mask | 1) as f32;
                let z_b: f32 = z0;
                value += (a_b * a_b)
                    * (a_b * a_b)
                    * Self::grad_coord_3d(
                        seed,
                        i + (!x_n_mask & Self::PRIME_X),
                        j + (!y_n_mask & Self::PRIME_Y),
                        k + (z_n_mask & Self::PRIME_Z),
                        x_b,
                        y_b,
                        z_b,
                    );
            }

            let a_c: f32 = z_a_flip_mask_1 + a1;
            if a_c > 0.0 {
                let x_c: f32 = x1;
                let y_c: f32 = y1;
                let z_c: f32 = (z_n_mask | 1) as f32 + z1;
                value += (a_c * a_c)
                    * (a_c * a_c)
                    * Self::grad_coord_3d(
                        seed2,
                        i.wrapping_add(Self::PRIME_X),
                        j.wrapping_add(Self::PRIME_Y),
                        k + (z_n_mask & (Self::PRIME_Z << 1)),
                        x_c,
                        y_c,
                        z_c,
                    );
                skip_d = true;
            }
        }

        if !skip_5 {
            let a5: f32 = y_a_flip_mask_1 + z_a_flip_mask_1 + a1;
            if a5 > 0.0 {
                let x5: f32 = x1;
                let y5: f32 = (y_n_mask | 1) as f32 + y1;
                let z5: f32 = (z_n_mask | 1) as f32 + z1;
                value += (a5 * a5)
                    * (a5 * a5)
                    * Self::grad_coord_3d(
                        seed2,
                        i.wrapping_add(Self::PRIME_X),
                        j + (y_n_mask & (Self::PRIME_Y << 1)),
                        k + (z_n_mask & (Self::PRIME_Z << 1)),
                        x5,
                        y5,
                        z5,
                    );
            }
        }

        if !skip_9 {
            let a9: f32 = x_a_flip_mask_1 + z_a_flip_mask_1 + a1;
            if a9 > 0.0 {
                let x9: f32 = (x_n_mask | 1) as f32 + x1;
                let y9: f32 = y1;
                let z9: f32 = (z_n_mask | 1) as f32 + z1;
                value += (a9 * a9)
                    * (a9 * a9)
                    * Self::grad_coord_3d(
                        seed2,
                        i + (x_n_mask & (Self::PRIME_X * 2)),
                        j.wrapping_add(Self::PRIME_Y),
                        k + (z_n_mask & (Self::PRIME_Z << 1)),
                        x9,
                        y9,
                        z9,
                    );
            }
        }

        if !skip_d {
            let a_d: f32 = x_a_flip_mask_1 + y_a_flip_mask_1 + a1;
            if a_d > 0.0 {
                let x_d: f32 = (x_n_mask | 1) as f32 + x1;
                let y_d: f32 = (y_n_mask | 1) as f32 + y1;
                let z_d: f32 = z1;
                value += (a_d * a_d)
                    * (a_d * a_d)
                    * Self::grad_coord_3d(
                        seed2,
                        i + (x_n_mask & (Self::PRIME_X << 1)),
                        j + (y_n_mask & (Self::PRIME_Y << 1)),
                        k.wrapping_add(Self::PRIME_Z),
                        x_d,
                        y_d,
                        z_d,
                    );
            }
        }

        return value * 9.046026385208288;
    }

    fn single_cellular_2d(&self, seed: i32, x: f32, y: f32) -> f32 {
        let xr: i32 = fast_round(x);
        let yr: i32 = fast_round(y);

        let mut distance0: f32 = 1e10;
        let mut distance1: f32 = 1e10;
        let mut closest_hash: i32 = 0;

        let cellular_jitter: f32 = 0.43701595 * self.cellular_jitter_modifier;

        let mut x_primed: i32 = (xr - 1) * Self::PRIME_X;
        let y_primed_base: i32 = (yr - 1) * Self::PRIME_Y;

        match self.cellular_distance_function {
            CellularDistanceFunction::Manhattan => {
                for xi in (xr - 1)..=(xr + 1) {
                    let mut y_primed: i32 = y_primed_base;

                    for yi in (yr - 1)..=(yr + 1) {
                        let hash: i32 = Self::hash_2d(seed, x_primed, y_primed);
                        let idx: i32 = hash & (255 << 1);

                        let vec_x: f32 =
                            (xi as f32 - x) + lookup::RAND_VECS_2D[idx as usize] * cellular_jitter;
                        let vec_y: f32 = (yi as f32 - y)
                            + lookup::RAND_VECS_2D[(idx | 1) as usize] * cellular_jitter;

                        let new_distance: f32 = fast_abs(vec_x) + fast_abs(vec_y);

                        distance1 = fast_max(fast_min(distance1, new_distance), distance0);
                        if new_distance < distance0 {
                            distance0 = new_distance;
                            closest_hash = hash;
                        }
                        y_primed += Self::PRIME_Y;
                    }
                    x_primed += Self::PRIME_X;
                }
            }
            CellularDistanceFunction::Hybrid => {
                for xi in (xr - 1)..=(xr + 1) {
                    let mut y_primed: i32 = y_primed_base;

                    for yi in (yr - 1)..=(yr + 1) {
                        let hash: i32 = Self::hash_2d(seed, x_primed, y_primed);
                        let idx: i32 = hash & (255 << 1);

                        let vec_x: f32 =
                            (xi as f32 - x) + lookup::RAND_VECS_2D[idx as usize] * cellular_jitter;
                        let vec_y: f32 = (yi as f32 - y)
                            + lookup::RAND_VECS_2D[(idx | 1) as usize] * cellular_jitter;

                        let new_distance: f32 =
                            (fast_abs(vec_x) + fast_abs(vec_y)) + (vec_x * vec_x + vec_y * vec_y);

                        distance1 = fast_max(fast_min(distance1, new_distance), distance0);
                        if new_distance < distance0 {
                            distance0 = new_distance;
                            closest_hash = hash;
                        }
                        y_primed += Self::PRIME_Y;
                    }
                    x_primed += Self::PRIME_X;
                }
            }
            _ => {
                for xi in (xr - 1)..=(xr + 1) {
                    let mut y_primed: i32 = y_primed_base;

                    for yi in (yr - 1)..=(yr + 1) {
                        let hash: i32 = Self::hash_2d(seed, x_primed, y_primed);
                        let idx: i32 = hash & (255 << 1);

                        let vec_x: f32 =
                            (xi as f32 - x) + lookup::RAND_VECS_2D[idx as usize] * cellular_jitter;
                        let vec_y: f32 = (yi as f32 - y)
                            + lookup::RAND_VECS_2D[(idx | 1) as usize] * cellular_jitter;

                        let new_distance: f32 = vec_x * vec_x + vec_y * vec_y;

                        distance1 = fast_max(fast_min(distance1, new_distance), distance0);
                        if new_distance < distance0 {
                            distance0 = new_distance;
                            closest_hash = hash;
                        }
                        y_primed += Self::PRIME_Y;
                    }
                    x_primed += Self::PRIME_X;
                }
            }
        }

        if self.cellular_distance_function == CellularDistanceFunction::Euclidean
            && self.cellular_return_type >= CellularReturnType::Distance
        {
            distance0 = fast_sqrt(distance0);

            if self.cellular_return_type >= CellularReturnType::Distance2 {
                distance1 = fast_sqrt(distance1);
            }
        }

        match self.cellular_return_type {
            CellularReturnType::CellValue => closest_hash as f32 * (1.0 / 2147483648.0),
            CellularReturnType::Distance => distance0 - 1.0,
            CellularReturnType::Distance2 => distance1 - 1.0,
            CellularReturnType::Distance2Add => (distance1 + distance0) * 0.5 - 1.0,
            CellularReturnType::Distance2Sub => distance1 - distance0 - 1.0,
            CellularReturnType::Distance2Mul => distance1 * distance0 * 0.5 - 1.0,
            CellularReturnType::Distance2Div => distance0 / distance1 - 1.0,
        }
    }

    fn single_cellular_3d(&self, seed: i32, x: f32, y: f32, z: f32) -> f32 {
        let xr: i32 = fast_round(x);
        let yr: i32 = fast_round(y);
        let zr: i32 = fast_round(z);

        let mut distance0: f32 = 1e10;
        let mut distance1: f32 = 1e10;
        let mut closest_hash: i32 = 0;

        let cellular_jitter: f32 = 0.39614353 * self.cellular_jitter_modifier;

        let mut x_primed: i32 = (xr - 1) * Self::PRIME_X;
        let y_primed_base: i32 = (yr - 1) * Self::PRIME_Y;
        let z_primed_base: i32 = (zr - 1) * Self::PRIME_Z;

        match self.cellular_distance_function {
            CellularDistanceFunction::Euclidean | CellularDistanceFunction::EuclideanSq => {
                for xi in (xr - 1)..=(xr + 1) {
                    let mut y_primed: i32 = y_primed_base;

                    for yi in (yr - 1)..=(yr + 1) {
                        let mut z_primed: i32 = z_primed_base;

                        for zi in (zr - 1)..=(zr + 1) {
                            let hash: i32 = Self::hash_3d(seed, x_primed, y_primed, z_primed);
                            let idx: i32 = hash & (255 << 2);

                            let vec_x: f32 = (xi as f32 - x)
                                + lookup::RAND_VECS_3D[idx as usize] * cellular_jitter;
                            let vec_y: f32 = (yi as f32 - y)
                                + lookup::RAND_VECS_3D[(idx | 1) as usize] * cellular_jitter;
                            let vec_z: f32 = (zi as f32 - z)
                                + lookup::RAND_VECS_3D[(idx | 2) as usize] * cellular_jitter;

                            let new_distance: f32 = vec_x * vec_x + vec_y * vec_y + vec_z * vec_z;

                            distance1 = fast_max(fast_min(distance1, new_distance), distance0);
                            if new_distance < distance0 {
                                distance0 = new_distance;
                                closest_hash = hash;
                            }
                            z_primed += Self::PRIME_Z;
                        }
                        y_primed += Self::PRIME_Y;
                    }
                    x_primed += Self::PRIME_X;
                }
            }
            CellularDistanceFunction::Manhattan => {
                for xi in (xr - 1)..=(xr + 1) {
                    let mut y_primed: i32 = y_primed_base;

                    for yi in (yr - 1)..=(yr + 1) {
                        let mut z_primed: i32 = z_primed_base;

                        for zi in (zr - 1)..=(zr + 1) {
                            let hash: i32 = Self::hash_3d(seed, x_primed, y_primed, z_primed);
                            let idx: i32 = hash & (255 << 2);

                            let vec_x: f32 = (xi as f32 - x)
                                + lookup::RAND_VECS_3D[idx as usize] * cellular_jitter;
                            let vec_y: f32 = (yi as f32 - y)
                                + lookup::RAND_VECS_3D[(idx | 1) as usize] * cellular_jitter;
                            let vec_z: f32 = (zi as f32 - z)
                                + lookup::RAND_VECS_3D[(idx | 2) as usize] * cellular_jitter;

                            let new_distance: f32 =
                                fast_abs(vec_x) + fast_abs(vec_y) + fast_abs(vec_z);

                            distance1 = fast_max(fast_min(distance1, new_distance), distance0);
                            if new_distance < distance0 {
                                distance0 = new_distance;
                                closest_hash = hash;
                            }
                            z_primed += Self::PRIME_Z;
                        }
                        y_primed += Self::PRIME_Y;
                    }
                    x_primed += Self::PRIME_X;
                }
            }
            CellularDistanceFunction::Hybrid => {
                for xi in (xr - 1)..=(xr + 1) {
                    let mut y_primed: i32 = y_primed_base;

                    for yi in (yr - 1)..=(yr + 1) {
                        let mut z_primed: i32 = z_primed_base;

                        for zi in (zr - 1)..=(zr + 1) {
                            let hash: i32 = Self::hash_3d(seed, x_primed, y_primed, z_primed);
                            let idx: i32 = hash & (255 << 2);

                            let vec_x: f32 = (xi as f32 - x)
                                + lookup::RAND_VECS_3D[idx as usize] * cellular_jitter;
                            let vec_y: f32 = (yi as f32 - y)
                                + lookup::RAND_VECS_3D[(idx | 1) as usize] * cellular_jitter;
                            let vec_z: f32 = (zi as f32 - z)
                                + lookup::RAND_VECS_3D[(idx | 2) as usize] * cellular_jitter;

                            let new_distance: f32 =
                                (fast_abs(vec_x) + fast_abs(vec_y) + fast_abs(vec_z))
                                    + (vec_x * vec_x + vec_y * vec_y + vec_z * vec_z);

                            distance1 = fast_max(fast_min(distance1, new_distance), distance0);
                            if new_distance < distance0 {
                                distance0 = new_distance;
                                closest_hash = hash;
                            }
                            z_primed += Self::PRIME_Z;
                        }
                        y_primed += Self::PRIME_Y;
                    }
                    x_primed += Self::PRIME_X;
                }
            }
        };

        if self.cellular_distance_function == CellularDistanceFunction::Euclidean
            && self.cellular_return_type >= CellularReturnType::Distance
        {
            distance0 = fast_sqrt(distance0);

            if self.cellular_return_type >= CellularReturnType::Distance2 {
                distance1 = fast_sqrt(distance1);
            }
        }

        match self.cellular_return_type {
            CellularReturnType::CellValue => closest_hash as f32 * (1.0 / 2147483648.0),
            CellularReturnType::Distance => distance0 - 1.0,
            CellularReturnType::Distance2 => distance1 - 1.0,
            CellularReturnType::Distance2Add => (distance1 + distance0) * 0.5 - 1.0,
            CellularReturnType::Distance2Sub => distance1 - distance0 - 1.0,
            CellularReturnType::Distance2Mul => distance1 * distance0 * 0.5 - 1.0,
            CellularReturnType::Distance2Div => distance0 / distance1 - 1.0,
        }
    }

    fn single_perlin_2d(&self, seed: i32, x: f32, y: f32) -> f32 {
        let mut x0: i32 = fast_floor(x);
        let mut y0: i32 = fast_floor(y);

        let xd0: f32 = x - x0 as f32;
        let yd0: f32 = y - y0 as f32;
        let xd1: f32 = xd0 - 1.0;
        let yd1: f32 = yd0 - 1.0;

        let xs: f32 = interp_quintic(xd0);
        let ys: f32 = interp_quintic(yd0);

        x0 *= Self::PRIME_X;
        y0 *= Self::PRIME_Y;
        let x1: i32 = x0.wrapping_add(Self::PRIME_X);
        let y1: i32 = y0.wrapping_add(Self::PRIME_Y);

        let xf0: f32 = lerp(
            Self::grad_coord_2d(seed, x0, y0, xd0, yd0),
            Self::grad_coord_2d(seed, x1, y0, xd1, yd0),
            xs,
        );
        let xf1: f32 = lerp(
            Self::grad_coord_2d(seed, x0, y1, xd0, yd1),
            Self::grad_coord_2d(seed, x1, y1, xd1, yd1),
            xs,
        );

        return lerp(xf0, xf1, ys) * 1.4247691104677813;
    }

    fn single_perlin_3d(&self, seed: i32, x: f32, y: f32, z: f32) -> f32 {
        let mut x0: i32 = fast_floor(x);
        let mut y0: i32 = fast_floor(y);
        let mut z0: i32 = fast_floor(z);

        let xd0: f32 = x - x0 as f32;
        let yd0: f32 = y - y0 as f32;
        let zd0: f32 = z - z0 as f32;
        let xd1: f32 = xd0 - 1.0;
        let yd1: f32 = yd0 - 1.0;
        let zd1: f32 = zd0 - 1.0;

        let xs: f32 = interp_quintic(xd0);
        let ys: f32 = interp_quintic(yd0);
        let zs: f32 = interp_quintic(zd0);

        x0 *= Self::PRIME_X;
        y0 *= Self::PRIME_Y;
        z0 *= Self::PRIME_Z;
        let x1: i32 = x0.wrapping_add(Self::PRIME_X);
        let y1: i32 = y0.wrapping_add(Self::PRIME_Y);
        let z1: i32 = z0.wrapping_add(Self::PRIME_Z);

        let xf00: f32 = lerp(
            Self::grad_coord_3d(seed, x0, y0, z0, xd0, yd0, zd0),
            Self::grad_coord_3d(seed, x1, y0, z0, xd1, yd0, zd0),
            xs,
        );
        let xf10: f32 = lerp(
            Self::grad_coord_3d(seed, x0, y1, z0, xd0, yd1, zd0),
            Self::grad_coord_3d(seed, x1, y1, z0, xd1, yd1, zd0),
            xs,
        );
        let xf01: f32 = lerp(
            Self::grad_coord_3d(seed, x0, y0, z1, xd0, yd0, zd1),
            Self::grad_coord_3d(seed, x1, y0, z1, xd1, yd0, zd1),
            xs,
        );
        let xf11: f32 = lerp(
            Self::grad_coord_3d(seed, x0, y1, z1, xd0, yd1, zd1),
            Self::grad_coord_3d(seed, x1, y1, z1, xd1, yd1, zd1),
            xs,
        );

        let yf0: f32 = lerp(xf00, xf10, ys);
        let yf1: f32 = lerp(xf01, xf11, ys);

        return lerp(yf0, yf1, zs) * 0.964921414852142333984375;
    }

    fn single_value_cubic_2d(&self, seed: i32, x: f32, y: f32) -> f32 {
        let mut x1: i32 = fast_floor(x);
        let mut y1: i32 = fast_floor(y);

        let xs: f32 = x - x1 as f32;
        let ys: f32 = y - y1 as f32;

        x1 *= Self::PRIME_X;
        y1 *= Self::PRIME_Y;
        let x0: i32 = x1.wrapping_sub(Self::PRIME_X);
        let y0: i32 = y1.wrapping_sub(Self::PRIME_Y);
        let x2: i32 = x1.wrapping_add(Self::PRIME_X);
        let y2: i32 = y1.wrapping_add(Self::PRIME_Y);
        let x3: i32 = x1 + ((Self::PRIME_X as i64) << 1) as i32;
        let y3: i32 = y1 + ((Self::PRIME_Y as i64) << 1) as i32;

        return cubic_lerp(
            cubic_lerp(
                Self::val_coord_2d(seed, x0, y0),
                Self::val_coord_2d(seed, x1, y0),
                Self::val_coord_2d(seed, x2, y0),
                Self::val_coord_2d(seed, x3, y0),
                xs,
            ),
            cubic_lerp(
                Self::val_coord_2d(seed, x0, y1),
                Self::val_coord_2d(seed, x1, y1),
                Self::val_coord_2d(seed, x2, y1),
                Self::val_coord_2d(seed, x3, y1),
                xs,
            ),
            cubic_lerp(
                Self::val_coord_2d(seed, x0, y2),
                Self::val_coord_2d(seed, x1, y2),
                Self::val_coord_2d(seed, x2, y2),
                Self::val_coord_2d(seed, x3, y2),
                xs,
            ),
            cubic_lerp(
                Self::val_coord_2d(seed, x0, y3),
                Self::val_coord_2d(seed, x1, y3),
                Self::val_coord_2d(seed, x2, y3),
                Self::val_coord_2d(seed, x3, y3),
                xs,
            ),
            ys,
        ) * (1.0 / (1.5 * 1.5));
    }

    fn single_value_cubic_3d(&self, seed: i32, x: f32, y: f32, z: f32) -> f32 {
        let mut x1: i32 = fast_floor(x);
        let mut y1: i32 = fast_floor(y);
        let mut z1: i32 = fast_floor(z);

        let xs: f32 = x - x1 as f32;
        let ys: f32 = y - y1 as f32;
        let zs: f32 = z - z1 as f32;

        x1 *= Self::PRIME_X;
        y1 *= Self::PRIME_Y;
        z1 *= Self::PRIME_Z;

        let x0: i32 = x1.wrapping_sub(Self::PRIME_X);
        let y0: i32 = y1.wrapping_sub(Self::PRIME_Y);
        let z0: i32 = z1.wrapping_sub(Self::PRIME_Z);
        let x2: i32 = x1.wrapping_add(Self::PRIME_X);
        let y2: i32 = y1.wrapping_add(Self::PRIME_Y);
        let z2: i32 = z1.wrapping_add(Self::PRIME_Z);
        let x3: i32 = x1 + ((Self::PRIME_X as i64) << 1) as i32;
        let y3: i32 = y1 + ((Self::PRIME_Y as i64) << 1) as i32;
        let z3: i32 = z1 + ((Self::PRIME_Z as i64) << 1) as i32;

        return cubic_lerp(
            cubic_lerp(
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y0, z0),
                    Self::val_coord_3d(seed, x1, y0, z0),
                    Self::val_coord_3d(seed, x2, y0, z0),
                    Self::val_coord_3d(seed, x3, y0, z0),
                    xs,
                ),
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y1, z0),
                    Self::val_coord_3d(seed, x1, y1, z0),
                    Self::val_coord_3d(seed, x2, y1, z0),
                    Self::val_coord_3d(seed, x3, y1, z0),
                    xs,
                ),
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y2, z0),
                    Self::val_coord_3d(seed, x1, y2, z0),
                    Self::val_coord_3d(seed, x2, y2, z0),
                    Self::val_coord_3d(seed, x3, y2, z0),
                    xs,
                ),
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y3, z0),
                    Self::val_coord_3d(seed, x1, y3, z0),
                    Self::val_coord_3d(seed, x2, y3, z0),
                    Self::val_coord_3d(seed, x3, y3, z0),
                    xs,
                ),
                ys,
            ),
            cubic_lerp(
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y0, z1),
                    Self::val_coord_3d(seed, x1, y0, z1),
                    Self::val_coord_3d(seed, x2, y0, z1),
                    Self::val_coord_3d(seed, x3, y0, z1),
                    xs,
                ),
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y1, z1),
                    Self::val_coord_3d(seed, x1, y1, z1),
                    Self::val_coord_3d(seed, x2, y1, z1),
                    Self::val_coord_3d(seed, x3, y1, z1),
                    xs,
                ),
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y2, z1),
                    Self::val_coord_3d(seed, x1, y2, z1),
                    Self::val_coord_3d(seed, x2, y2, z1),
                    Self::val_coord_3d(seed, x3, y2, z1),
                    xs,
                ),
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y3, z1),
                    Self::val_coord_3d(seed, x1, y3, z1),
                    Self::val_coord_3d(seed, x2, y3, z1),
                    Self::val_coord_3d(seed, x3, y3, z1),
                    xs,
                ),
                ys,
            ),
            cubic_lerp(
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y0, z2),
                    Self::val_coord_3d(seed, x1, y0, z2),
                    Self::val_coord_3d(seed, x2, y0, z2),
                    Self::val_coord_3d(seed, x3, y0, z2),
                    xs,
                ),
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y1, z2),
                    Self::val_coord_3d(seed, x1, y1, z2),
                    Self::val_coord_3d(seed, x2, y1, z2),
                    Self::val_coord_3d(seed, x3, y1, z2),
                    xs,
                ),
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y2, z2),
                    Self::val_coord_3d(seed, x1, y2, z2),
                    Self::val_coord_3d(seed, x2, y2, z2),
                    Self::val_coord_3d(seed, x3, y2, z2),
                    xs,
                ),
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y3, z2),
                    Self::val_coord_3d(seed, x1, y3, z2),
                    Self::val_coord_3d(seed, x2, y3, z2),
                    Self::val_coord_3d(seed, x3, y3, z2),
                    xs,
                ),
                ys,
            ),
            cubic_lerp(
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y0, z3),
                    Self::val_coord_3d(seed, x1, y0, z3),
                    Self::val_coord_3d(seed, x2, y0, z3),
                    Self::val_coord_3d(seed, x3, y0, z3),
                    xs,
                ),
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y1, z3),
                    Self::val_coord_3d(seed, x1, y1, z3),
                    Self::val_coord_3d(seed, x2, y1, z3),
                    Self::val_coord_3d(seed, x3, y1, z3),
                    xs,
                ),
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y2, z3),
                    Self::val_coord_3d(seed, x1, y2, z3),
                    Self::val_coord_3d(seed, x2, y2, z3),
                    Self::val_coord_3d(seed, x3, y2, z3),
                    xs,
                ),
                cubic_lerp(
                    Self::val_coord_3d(seed, x0, y3, z3),
                    Self::val_coord_3d(seed, x1, y3, z3),
                    Self::val_coord_3d(seed, x2, y3, z3),
                    Self::val_coord_3d(seed, x3, y3, z3),
                    xs,
                ),
                ys,
            ),
            zs,
        ) * (1.0 / (1.5 * 1.5 * 1.5));
    }

    fn single_value_2d(&self, seed: i32, x: f32, y: f32) -> f32 {
        let mut x0: i32 = fast_floor(x);
        let mut y0: i32 = fast_floor(y);

        let xs: f32 = interp_hermite(x - x0 as f32);
        let ys: f32 = interp_hermite(y - y0 as f32);

        x0 *= Self::PRIME_X;
        y0 *= Self::PRIME_Y;
        let x1: i32 = x0.wrapping_add(Self::PRIME_X);
        let y1: i32 = y0.wrapping_add(Self::PRIME_Y);

        let xf0: f32 = lerp(
            Self::val_coord_2d(seed, x0, y0),
            Self::val_coord_2d(seed, x1, y0),
            xs,
        );
        let xf1: f32 = lerp(
            Self::val_coord_2d(seed, x0, y1),
            Self::val_coord_2d(seed, x1, y1),
            xs,
        );

        return lerp(xf0, xf1, ys);
    }

    fn single_value_3d(&self, seed: i32, x: f32, y: f32, z: f32) -> f32 {
        let mut x0: i32 = fast_floor(x);
        let mut y0: i32 = fast_floor(y);
        let mut z0: i32 = fast_floor(z);

        let xs: f32 = interp_hermite(x - x0 as f32);
        let ys: f32 = interp_hermite(y - y0 as f32);
        let zs: f32 = interp_hermite(z - z0 as f32);

        x0 *= Self::PRIME_X;
        y0 *= Self::PRIME_Y;
        z0 *= Self::PRIME_Z;
        let x1: i32 = x0.wrapping_add(Self::PRIME_X);
        let y1: i32 = y0.wrapping_add(Self::PRIME_Y);
        let z1: i32 = z0.wrapping_add(Self::PRIME_Z);

        let xf00: f32 = lerp(
            Self::val_coord_3d(seed, x0, y0, z0),
            Self::val_coord_3d(seed, x1, y0, z0),
            xs,
        );
        let xf10: f32 = lerp(
            Self::val_coord_3d(seed, x0, y1, z0),
            Self::val_coord_3d(seed, x1, y1, z0),
            xs,
        );
        let xf01: f32 = lerp(
            Self::val_coord_3d(seed, x0, y0, z1),
            Self::val_coord_3d(seed, x1, y0, z1),
            xs,
        );
        let xf11: f32 = lerp(
            Self::val_coord_3d(seed, x0, y1, z1),
            Self::val_coord_3d(seed, x1, y1, z1),
            xs,
        );

        let yf0: f32 = lerp(xf00, xf10, ys);
        let yf1: f32 = lerp(xf01, xf11, ys);

        return lerp(yf0, yf1, zs);
    }

    fn do_single_domain_warp_2d(
        &self,
        seed: i32,
        amp: f32,
        freq: f32,
        x: f32,
        y: f32,
        xr: &mut f32,
        yr: &mut f32,
    ) {
        match self.domain_warp_type {
            DomainWarpType::OpenSimplex2 => self.single_domain_warp_simplex_gradient_2d(
                seed,
                amp * 38.283687591552734375,
                freq,
                x,
                y,
                xr,
                yr,
                false,
            ),
            DomainWarpType::OpenSimplex2Reduced => self.single_domain_warp_simplex_gradient_2d(
                seed,
                amp * 16.0,
                freq,
                x,
                y,
                xr,
                yr,
                true,
            ),
            DomainWarpType::BasicGrid => {
                self.single_domain_warp_basic_grid_2d(seed, amp, freq, x, y, xr, yr)
            }
        }
    }

    fn do_single_domain_warp_3d(
        &self,
        seed: i32,
        amp: f32,
        freq: f32,
        x: f32,
        y: f32,
        z: f32,
        xr: &mut f32,
        yr: &mut f32,
        zr: &mut f32,
    ) {
        match self.domain_warp_type {
            DomainWarpType::OpenSimplex2 => self.single_domain_warp_open_simplex2_gradient_3d(
                seed,
                amp * 32.69428253173828125,
                freq,
                x,
                y,
                z,
                xr,
                yr,
                zr,
                false,
            ),
            DomainWarpType::OpenSimplex2Reduced => self
                .single_domain_warp_open_simplex2_gradient_3d(
                    seed,
                    amp * 7.71604938271605,
                    freq,
                    x,
                    y,
                    z,
                    xr,
                    yr,
                    zr,
                    true,
                ),
            DomainWarpType::BasicGrid => {
                self.single_domain_warp_basic_grid_3d(seed, amp, freq, x, y, z, xr, yr, zr)
            }
        }
    }

    fn domain_warp_single_2d(&self, x: &mut f32, y: &mut f32) {
        let seed: i32 = self.seed;
        let amp: f32 = self.domain_warp_amp * self.fractal_bounding;
        let freq: f32 = self.frequency;

        let mut xs: f32 = *x;
        let mut ys: f32 = *y;
        self.transform_domain_warp_coordinate_2d(&mut xs, &mut ys);

        self.do_single_domain_warp_2d(seed, amp, freq, xs, ys, x, y);
    }

    fn domain_warp_single_3d(&self, x: &mut f32, y: &mut f32, z: &mut f32) {
        let seed: i32 = self.seed;
        let amp: f32 = self.domain_warp_amp * self.fractal_bounding;
        let freq: f32 = self.frequency;

        let mut xs: f32 = *x;
        let mut ys: f32 = *y;
        let mut zs: f32 = *z;
        self.transform_domain_warp_coordinate_3d(&mut xs, &mut ys, &mut zs);

        self.do_single_domain_warp_3d(seed, amp, freq, xs, ys, zs, x, y, z);
    }

    fn domain_warp_fractal_progressive_2d(&self, x: &mut f32, y: &mut f32) {
        let mut seed: i32 = self.seed;
        let mut amp: f32 = self.domain_warp_amp * self.fractal_bounding;
        let mut freq: f32 = self.frequency;

        for _ in 0..self.octaves {
            let mut xs: f32 = *x;
            let mut ys: f32 = *y;
            self.transform_domain_warp_coordinate_2d(&mut xs, &mut ys);

            self.do_single_domain_warp_2d(seed, amp, freq, xs, ys, x, y);

            seed += 1;
            amp *= self.gain;
            freq *= self.lacunarity;
        }
    }

    fn domain_warp_fractal_progressive_3d(&self, x: &mut f32, y: &mut f32, z: &mut f32) {
        let mut seed: i32 = self.seed;
        let mut amp: f32 = self.domain_warp_amp * self.fractal_bounding;
        let mut freq: f32 = self.frequency;

        for _ in 0..self.octaves {
            let mut xs: f32 = *x;
            let mut ys: f32 = *y;
            let mut zs: f32 = *z;
            self.transform_domain_warp_coordinate_3d(&mut xs, &mut ys, &mut zs);

            self.do_single_domain_warp_3d(seed, amp, freq, xs, ys, zs, x, y, z);

            seed += 1;
            amp *= self.gain;
            freq *= self.lacunarity;
        }
    }

    fn domain_warp_fractal_independent_2d(&self, x: &mut f32, y: &mut f32) {
        let mut xs: f32 = *x;
        let mut ys: f32 = *y;
        self.transform_domain_warp_coordinate_2d(&mut xs, &mut ys);

        let mut seed: i32 = self.seed;
        let mut amp: f32 = self.domain_warp_amp * self.fractal_bounding;
        let mut freq: f32 = self.frequency;

        for _ in 0..self.octaves {
            self.do_single_domain_warp_2d(seed, amp, freq, xs, ys, x, y);

            seed += 1;
            amp *= self.gain;
            freq *= self.lacunarity;
        }
    }

    fn domain_warp_fractal_independent_3d(&self, x: &mut f32, y: &mut f32, z: &mut f32) {
        let mut xs: f32 = *x;
        let mut ys: f32 = *y;
        let mut zs: f32 = *z;
        self.transform_domain_warp_coordinate_3d(&mut xs, &mut ys, &mut zs);

        let mut seed: i32 = self.seed;
        let mut amp: f32 = self.domain_warp_amp * self.fractal_bounding;
        let mut freq: f32 = self.frequency;

        for _ in 0..self.octaves {
            self.do_single_domain_warp_3d(seed, amp, freq, xs, ys, zs, x, y, z);

            seed += 1;
            amp *= self.gain;
            freq *= self.lacunarity;
        }
    }

    fn single_domain_warp_basic_grid_2d(
        &self,
        seed: i32,
        warp_amp: f32,
        frequency: f32,
        x: f32,
        y: f32,
        xr: &mut f32,
        yr: &mut f32,
    ) {
        let xf: f32 = x * frequency;
        let yf: f32 = y * frequency;

        let mut x0: i32 = fast_floor(xf);
        let mut y0: i32 = fast_floor(yf);

        let xs: f32 = interp_hermite(xf - x0 as f32);
        let ys: f32 = interp_hermite(yf - y0 as f32);

        x0 *= Self::PRIME_X;
        y0 *= Self::PRIME_Y;
        let x1: i32 = x0.wrapping_add(Self::PRIME_X);
        let y1: i32 = y0.wrapping_add(Self::PRIME_Y);

        let mut hash0: i32 = Self::hash_2d(seed, x0, y0) & (255 << 1);
        let mut hash1: i32 = Self::hash_2d(seed, x1, y0) & (255 << 1);

        let lx0x: f32 = lerp(
            lookup::RAND_VECS_2D[hash0 as usize],
            lookup::RAND_VECS_2D[hash1 as usize],
            xs,
        );
        let ly0x: f32 = lerp(
            lookup::RAND_VECS_2D[(hash0 | 1) as usize],
            lookup::RAND_VECS_2D[(hash1 | 1) as usize],
            xs,
        );

        hash0 = Self::hash_2d(seed, x0, y1) & (255 << 1);
        hash1 = Self::hash_2d(seed, x1, y1) & (255 << 1);

        let lx1x: f32 = lerp(
            lookup::RAND_VECS_2D[hash0 as usize],
            lookup::RAND_VECS_2D[hash1 as usize],
            xs,
        );
        let ly1x: f32 = lerp(
            lookup::RAND_VECS_2D[(hash0 | 1) as usize],
            lookup::RAND_VECS_2D[(hash1 | 1) as usize],
            xs,
        );

        *xr += lerp(lx0x, lx1x, ys) * warp_amp;
        *yr += lerp(ly0x, ly1x, ys) * warp_amp;
    }

    fn single_domain_warp_basic_grid_3d(
        &self,
        seed: i32,
        warp_amp: f32,
        frequency: f32,
        x: f32,
        y: f32,
        z: f32,
        xr: &mut f32,
        yr: &mut f32,
        zr: &mut f32,
    ) {
        let xf: f32 = x * frequency;
        let yf: f32 = y * frequency;
        let zf: f32 = z * frequency;

        let mut x0: i32 = fast_floor(xf);
        let mut y0: i32 = fast_floor(yf);
        let mut z0: i32 = fast_floor(zf);

        let xs: f32 = interp_hermite(xf - x0 as f32);
        let ys: f32 = interp_hermite(yf - y0 as f32);
        let zs: f32 = interp_hermite(zf - z0 as f32);

        x0 *= Self::PRIME_X;
        y0 *= Self::PRIME_Y;
        z0 *= Self::PRIME_Z;
        let x1: i32 = x0.wrapping_add(Self::PRIME_X);
        let y1: i32 = y0.wrapping_add(Self::PRIME_Y);
        let z1: i32 = z0.wrapping_add(Self::PRIME_Z);

        let mut hash0: i32 = Self::hash_3d(seed, x0, y0, z0) & (255 << 2);
        let mut hash1: i32 = Self::hash_3d(seed, x1, y0, z0) & (255 << 2);

        let mut lx0x: f32 = lerp(
            lookup::RAND_VECS_3D[hash0 as usize],
            lookup::RAND_VECS_3D[hash1 as usize],
            xs,
        );
        let mut ly0x: f32 = lerp(
            lookup::RAND_VECS_3D[(hash0 | 1) as usize],
            lookup::RAND_VECS_3D[(hash1 | 1) as usize],
            xs,
        );
        let mut lz0x: f32 = lerp(
            lookup::RAND_VECS_3D[(hash0 | 2) as usize],
            lookup::RAND_VECS_3D[(hash1 | 2) as usize],
            xs,
        );

        hash0 = Self::hash_3d(seed, x0, y1, z0) & (255 << 2);
        hash1 = Self::hash_3d(seed, x1, y1, z0) & (255 << 2);

        let mut lx1x: f32 = lerp(
            lookup::RAND_VECS_3D[hash0 as usize],
            lookup::RAND_VECS_3D[hash1 as usize],
            xs,
        );
        let mut ly1x: f32 = lerp(
            lookup::RAND_VECS_3D[(hash0 | 1) as usize],
            lookup::RAND_VECS_3D[(hash1 | 1) as usize],
            xs,
        );
        let mut lz1x: f32 = lerp(
            lookup::RAND_VECS_3D[(hash0 | 2) as usize],
            lookup::RAND_VECS_3D[(hash1 | 2) as usize],
            xs,
        );

        let lx0y: f32 = lerp(lx0x, lx1x, ys);
        let ly0y: f32 = lerp(ly0x, ly1x, ys);
        let lz0y: f32 = lerp(lz0x, lz1x, ys);

        hash0 = Self::hash_3d(seed, x0, y0, z1) & (255 << 2);
        hash1 = Self::hash_3d(seed, x1, y0, z1) & (255 << 2);

        lx0x = lerp(
            lookup::RAND_VECS_3D[hash0 as usize],
            lookup::RAND_VECS_3D[hash1 as usize],
            xs,
        );
        ly0x = lerp(
            lookup::RAND_VECS_3D[(hash0 | 1) as usize],
            lookup::RAND_VECS_3D[(hash1 | 1) as usize],
            xs,
        );
        lz0x = lerp(
            lookup::RAND_VECS_3D[(hash0 | 2) as usize],
            lookup::RAND_VECS_3D[(hash1 | 2) as usize],
            xs,
        );

        hash0 = Self::hash_3d(seed, x0, y1, z1) & (255 << 2);
        hash1 = Self::hash_3d(seed, x1, y1, z1) & (255 << 2);

        lx1x = lerp(
            lookup::RAND_VECS_3D[hash0 as usize],
            lookup::RAND_VECS_3D[hash1 as usize],
            xs,
        );
        ly1x = lerp(
            lookup::RAND_VECS_3D[(hash0 | 1) as usize],
            lookup::RAND_VECS_3D[(hash1 | 1) as usize],
            xs,
        );
        lz1x = lerp(
            lookup::RAND_VECS_3D[(hash0 | 2) as usize],
            lookup::RAND_VECS_3D[(hash1 | 2) as usize],
            xs,
        );

        *xr += lerp(lx0y, lerp(lx0x, lx1x, ys), zs) * warp_amp;
        *yr += lerp(ly0y, lerp(ly0x, ly1x, ys), zs) * warp_amp;
        *zr += lerp(lz0y, lerp(lz0x, lz1x, ys), zs) * warp_amp;
    }

    fn single_domain_warp_simplex_gradient_2d(
        &self,
        seed: i32,
        warp_amp: f32,
        frequency: f32,
        mut x: f32,
        mut y: f32,
        xr: &mut f32,
        yr: &mut f32,
        out_grad_only: bool,
    ) {
        const SQRT3: f32 = 1.7320508075688772935274463415059;
        const G2: f32 = (3.0 - SQRT3) / 6.0;

        x *= frequency;
        y *= frequency;

        /*
         * --- Skew moved to TransformNoiseCoordinate method ---
         * const F2: f32 = 0.5f * (SQRT3 - 1);
         * s: f32 = (x + y) * F2;
         * x += s; y += s;
         */

        let mut i: i32 = fast_floor(x);
        let mut j: i32 = fast_floor(y);
        let xi: f32 = x - i as f32;
        let yi: f32 = y - j as f32;

        let t: f32 = (xi + yi) * G2;
        let x0: f32 = xi - t;
        let y0: f32 = yi - t;

        i = i.wrapping_mul(Self::PRIME_X);
        j = j.wrapping_mul(Self::PRIME_Y);

        let mut vx: f32 = 0.0;
        let mut vy: f32 = 0.0;

        let a: f32 = 0.5 - x0 * x0 - y0 * y0;
        if a > 0.0 {
            let aaaa: f32 = (a * a) * (a * a);
            let mut xo: f32 = 0.0;
            let mut yo: f32 = 0.0;
            if out_grad_only {
                Self::grad_coord_out_2d(seed, i, j, &mut xo, &mut yo);
            } else {
                Self::grad_coord_dual_2d(seed, i, j, x0, y0, &mut xo, &mut yo);
            }
            vx += aaaa * xo;
            vy += aaaa * yo;
        }

        let c: f32 = (2.0 * (1.0 - 2.0 * G2) * (1.0 / G2 - 2.0)) * t
            + ((-2.0 * (1.0 - 2.0 * G2) * (1.0 - 2.0 * G2)) + a);
        if c > 0.0 {
            let x2: f32 = x0 + (2.0 * G2 - 1.0);
            let y2: f32 = y0 + (2.0 * G2 - 1.0);
            let cccc: f32 = (c * c) * (c * c);
            let mut xo: f32 = 0.0;
            let mut yo: f32 = 0.0;
            if out_grad_only {
                Self::grad_coord_out_2d(
                    seed,
                    i.wrapping_add(Self::PRIME_X),
                    j.wrapping_add(Self::PRIME_Y),
                    &mut xo,
                    &mut yo,
                );
            } else {
                Self::grad_coord_dual_2d(
                    seed,
                    i.wrapping_add(Self::PRIME_X),
                    j.wrapping_add(Self::PRIME_Y),
                    x2,
                    y2,
                    &mut xo,
                    &mut yo,
                );
            }
            vx += cccc * xo;
            vy += cccc * yo;
        }

        if y0 > x0 {
            let x1: f32 = x0 + G2;
            let y1: f32 = y0 + (G2 - 1.0);
            let b: f32 = 0.5 - x1 * x1 - y1 * y1;
            if b > 0.0 {
                let bbbb: f32 = (b * b) * (b * b);
                let mut xo: f32 = 0.0;
                let mut yo: f32 = 0.0;
                if out_grad_only {
                    Self::grad_coord_out_2d(
                        seed,
                        i,
                        j.wrapping_add(Self::PRIME_Y),
                        &mut xo,
                        &mut yo,
                    );
                } else {
                    Self::grad_coord_dual_2d(
                        seed,
                        i,
                        j.wrapping_add(Self::PRIME_Y),
                        x1,
                        y1,
                        &mut xo,
                        &mut yo,
                    );
                }
                vx += bbbb * xo;
                vy += bbbb * yo;
            }
        } else {
            let x1: f32 = x0 + (G2 - 1.0);
            let y1: f32 = y0 + G2;
            let b: f32 = 0.5 - x1 * x1 - y1 * y1;
            if b > 0.0 {
                let bbbb: f32 = (b * b) * (b * b);
                let mut xo: f32 = 0.0;
                let mut yo: f32 = 0.0;
                if out_grad_only {
                    Self::grad_coord_out_2d(
                        seed,
                        i.wrapping_add(Self::PRIME_X),
                        j,
                        &mut xo,
                        &mut yo,
                    );
                } else {
                    Self::grad_coord_dual_2d(
                        seed,
                        i.wrapping_add(Self::PRIME_X),
                        j,
                        x1,
                        y1,
                        &mut xo,
                        &mut yo,
                    );
                }
                vx += bbbb * xo;
                vy += bbbb * yo;
            }
        }

        *xr += vx * warp_amp;
        *yr += vy * warp_amp;
    }

    fn single_domain_warp_open_simplex2_gradient_3d(
        &self,
        mut seed: i32,
        warp_amp: f32,
        frequency: f32,
        mut x: f32,
        mut y: f32,
        mut z: f32,
        xr: &mut f32,
        yr: &mut f32,
        zr: &mut f32,
        out_grad_only: bool,
    ) {
        x *= frequency;
        y *= frequency;
        z *= frequency;

        /*
         * --- Rotation moved to TransformDomainWarpCoordinate method ---
         * const R3: f32 = (FNfloat)(2.0 / 3.0);
         * r: f32 = (x + y + z) * R3; // Rotation, not skew
         * x = r - x; y = r - y; z = r - z;
         */

        let mut i: i32 = fast_round(x);
        let mut j: i32 = fast_round(y);
        let mut k: i32 = fast_round(z);
        let mut x0: f32 = x - i as f32;
        let mut y0: f32 = y - j as f32;
        let mut z0: f32 = z - k as f32;

        let mut xn_sign: i32 = (-x0 - 1.0) as i32 | 1;
        let mut yn_sign: i32 = (-y0 - 1.0) as i32 | 1;
        let mut zn_sign: i32 = (-z0 - 1.0) as i32 | 1;

        let mut ax0: f32 = xn_sign as f32 * -x0;
        let mut ay0: f32 = yn_sign as f32 * -y0;
        let mut az0: f32 = zn_sign as f32 * -z0;

        i = i.wrapping_mul(Self::PRIME_X);
        j = j.wrapping_mul(Self::PRIME_Y);
        k = k.wrapping_mul(Self::PRIME_Z);

        let mut vx: f32 = 0.0;
        let mut vy: f32 = 0.0;
        let mut vz: f32 = 0.0;

        let mut a: f32 = (0.6 - x0 * x0) - (y0 * y0 + z0 * z0);

        for l in 0..2 {
            if a > 0.0 {
                let aaaa: f32 = (a * a) * (a * a);
                let mut xo: f32 = 0.0;
                let mut yo: f32 = 0.0;
                let mut zo: f32 = 0.0;
                if out_grad_only {
                    Self::grad_coord_out_3d(seed, i, j, k, &mut xo, &mut yo, &mut zo);
                } else {
                    Self::grad_coord_dual_3d(seed, i, j, k, x0, y0, z0, &mut xo, &mut yo, &mut zo);
                }
                vx += aaaa * xo;
                vy += aaaa * yo;
                vz += aaaa * zo;
            }

            let mut b: f32 = a + 1.0;
            let mut i1: i32 = i;
            let mut j1: i32 = j;
            let mut k1: i32 = k;
            let mut x1: f32 = x0;
            let mut y1: f32 = y0;
            let mut z1: f32 = z0;

            if ax0 >= ay0 && ax0 >= az0 {
                x1 += xn_sign as f32;
                b -= xn_sign as f32 * 2.0 * x1;
                i1 -= xn_sign * Self::PRIME_X;
            } else if ay0 > ax0 && ay0 >= az0 {
                y1 += yn_sign as f32;
                b -= yn_sign as f32 * 2.0 * y1;
                j1 -= yn_sign * Self::PRIME_Y;
            } else {
                z1 += zn_sign as f32;
                b -= zn_sign as f32 * 2.0 * z1;
                k1 -= zn_sign * Self::PRIME_Z;
            }

            if b > 0.0 {
                let bbbb: f32 = (b * b) * (b * b);
                let mut xo: f32 = 0.0;
                let mut yo: f32 = 0.0;
                let mut zo: f32 = 0.0;
                if out_grad_only {
                    Self::grad_coord_out_3d(seed, i1, j1, k1, &mut xo, &mut yo, &mut zo);
                } else {
                    Self::grad_coord_dual_3d(
                        seed, i1, j1, k1, x1, y1, z1, &mut xo, &mut yo, &mut zo,
                    );
                }
                vx += bbbb * xo;
                vy += bbbb * yo;
                vz += bbbb * zo;
            }

            if l == 1 {
                break;
            }

            ax0 = 0.5 - ax0;
            ay0 = 0.5 - ay0;
            az0 = 0.5 - az0;

            x0 = xn_sign as f32 * ax0;
            y0 = yn_sign as f32 * ay0;
            z0 = zn_sign as f32 * az0;

            a += (0.75 - ax0) - (ay0 + az0);

            i += (xn_sign >> 1) & Self::PRIME_X;
            j += (yn_sign >> 1) & Self::PRIME_Y;
            k += (zn_sign >> 1) & Self::PRIME_Z;

            xn_sign = -xn_sign;
            yn_sign = -yn_sign;
            zn_sign = -zn_sign;

            seed += 1293373;
        }

        *xr += vx * warp_amp;
        *yr += vy * warp_amp;
        *zr += vz * warp_amp;
    }
}
