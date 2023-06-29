use fast_noise_lite_rs::{FastNoiseLite, NoiseType, FractalType, DomainWarpType};

fn main() {
    let mut noise = FastNoiseLite::new(1337);

    noise.set_noise_type(NoiseType::OpenSimplex2);
    noise.set_frequency(0.02);
    noise.set_fractal_type(FractalType::DomainWarpProgressive);

    noise.set_domain_warp_amp(32.0);
    noise.set_domain_warp_type(DomainWarpType::OpenSimplex2);

    let mut x = 2.5;
    let mut y = 6.2;
    noise.domain_warp_2d(&mut x, &mut y);
    let noise_value = noise.get_noise_2d(x, y);

    println!("{noise_value:?}");
}
