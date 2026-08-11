// Blade port of inkwell-webgpu-water (open-ocean, optimized, surface view).
//
// WGSL taken from src/lib/webgpu-water-engine.ts with these changes:
// - all @group/@binding annotations removed: blade assigns bindings from the
//   ShaderData struct field order (see engine.rs)
// - global variables renamed to snake_case to match the Rust ShaderData fields
// - template-interpolated constants inlined (cascade scales/choppiness, clipmap
//   resolution, breaker event resolution, water level thresholds)
// - shore-scene capture block removed from waterFragment (open-ocean build)

struct WorldUniforms {
  viewProj: mat4x4<f32>,
  cameraTime: vec4<f32>,
  cameraRight: vec4<f32>,
  cameraUp: vec4<f32>,
  cameraForward: vec4<f32>,
  sunWater: vec4<f32>,
  terrain: vec4<f32>,
  simulation: vec4<f32>,
  player: vec4<f32>,
  interaction: vec4<f32>,
  environment: vec4<f32>,
}
var<uniform> uniforms: WorldUniforms;

fn hash21(p: vec2<f32>) -> f32 {
  var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
  p3 += vec3<f32>(dot(p3, p3.yzx + vec3<f32>(33.33)));
  return fract((p3.x + p3.y) * p3.z);
}

fn valueNoise(p: vec2<f32>) -> f32 {
  let cell = floor(p);
  var local = fract(p);
  local = local * local * (vec2<f32>(3.0) - 2.0 * local);
  return mix(
    mix(hash21(cell), hash21(cell + vec2<f32>(1.0, 0.0)), local.x),
    mix(hash21(cell + vec2<f32>(0.0, 1.0)), hash21(cell + vec2<f32>(1.0, 1.0)), local.x),
    local.y
  );
}

fn tethysCoastalShelf(p: vec2<f32>, center: vec2<f32>, radiusScale: vec2<f32>, lift: f32, relief: f32, phase: f32) -> f32 {
  let delta = p - center;
  let angle = phase * 0.23;
  let local = vec2<f32>(
    delta.x * cos(angle) - delta.y * sin(angle),
    delta.x * sin(angle) + delta.y * cos(angle)
  ) / radiusScale;
  let radius = length(local);
  let coastAngle = atan2(local.y, local.x);
  let coastNoise = sin(coastAngle * 3.0 + phase) * 0.040
    + sin(coastAngle * 7.0 - phase * 0.8) * 0.022
    + sin((p.x + p.y) * 0.031 + phase) * 0.018;
  let coastalDistance = radius + coastNoise;
  let coast = 1.0 - smoothstep(0.58, 1.035, coastalDistance);
  let interior = 1.0 - smoothstep(0.12, 0.57, coastalDistance);
  let erosion = sin(p.x * 0.038 + p.y * 0.017 + phase) * 0.52
    + sin(p.x * -0.019 + p.y * 0.043 - phase * 0.7) * 0.31
    + sin((p.x + p.y) * 0.081 + phase * 1.4) * 0.17;
  let longRidge = sin(p.x * 0.014 - p.y * 0.021 + phase * 2.1);
  let highland = pow(smoothstep(-0.20, 0.85, longRidge), 1.35);
  let rollingRelief = -0.08 + erosion * 0.22 + highland * 0.86;
  return max(0.0, coast * (lift + relief * rollingRelief * interior));
}

fn terrainHeight(p: vec2<f32>, shoreMix: f32) -> f32 {
  var warped = p;
  warped.x += sin(p.y * 0.018 + 0.8) * 2.4;
  warped.y += sin(p.x * 0.016 - 0.2) * 2.1;
  var height = -8.5;
  height += sin(warped.x * 0.052 + warped.y * 0.016) * 0.62;
  height += sin(warped.x * -0.024 + warped.y * 0.046 + 1.7) * 0.39;
  height += sin((warped.x + warped.y) * 0.12) * 0.14;
  var shelfPower = 0.0;
  shelfPower += pow(tethysCoastalShelf(warped, vec2<f32>(0.0, 14.0), vec2<f32>(76.0, 50.0), 12.8, 8.0, 0.3), 6.0);
  shelfPower += pow(tethysCoastalShelf(warped, vec2<f32>(-112.0, -79.0), vec2<f32>(62.0, 40.0), 12.6, 11.0, 1.7), 6.0);
  shelfPower += pow(tethysCoastalShelf(warped, vec2<f32>(116.0, -92.0), vec2<f32>(65.0, 44.0), 12.5, 12.0, 3.4), 6.0);
  shelfPower += pow(tethysCoastalShelf(warped, vec2<f32>(-6.0, -196.0), vec2<f32>(112.0, 40.0), 12.9, 10.0, 5.1), 6.0);
  height += pow(max(shelfPower, 0.0), 1.0 / 6.0);
  // Build broad, domain-warped dune ridges inland. Keeping their mask above
  // the swash zone protects the waterline contour while giving the exposed
  // islands a wind-shaped silhouette instead of a smooth clay mound.
  let duneInterior = smoothstep(2.02, 3.95, height);
  let duneWarp = vec2<f32>(
    valueNoise(p * 0.021 + vec2<f32>(7.1, -3.8)) - 0.5,
    valueNoise(p * 0.024 + vec2<f32>(-5.3, 9.6)) - 0.5
  );
  let duneP = p + duneWarp * 17.0;
  let duneBands = sin(duneP.x * 0.098 + duneP.y * 0.031)
    + sin(duneP.x * 0.047 - duneP.y * 0.071 + 1.8) * 0.47;
  let duneRidges = sign(duneBands) * pow(abs(duneBands) * 0.68, 1.32);
  let broadDunes = valueNoise(duneP * 0.038 + vec2<f32>(2.3, 6.7)) - 0.5;
  let erodedDetail = valueNoise(duneP * 0.14 + vec2<f32>(4.7, -2.1)) - 0.5;
  height += shoreMix * duneInterior
    * (duneRidges * 0.52 + broadDunes * 0.82 + erodedDetail * 0.16);
  // This lab isolates the water material. Preserve Tethys' shelf contours as
  // a submerged seabed, but never expose an island or terrestrial surface.
  var seabed = min(height, -4.35);
  seabed += sin(p.x * 0.071 + p.y * 0.026) * 0.38;
  seabed += sin(p.x * -0.033 + p.y * 0.083 + 1.7) * 0.24;
  seabed += sin(p.x * 0.017 - p.y * 0.013 + 0.6) * 0.48;
  // The material lab keeps the original all-submerged view, while the coastal
  // scene restores authored Tethys islands for wet/dry and run-up validation.
  return mix(seabed, height, shoreMix);
}

// ---------------------------------------------------------------------------
// Terrain field compute
// ---------------------------------------------------------------------------
var field_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn buildTerrain(@builtin(global_invocation_id) id: vec3<u32>) {
  let dimensions = textureDimensions(field_out);
  if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }
  let uv = vec2<f32>(id.xy) / vec2<f32>(dimensions - vec2<u32>(1u));
  let p = (uv - vec2<f32>(0.5)) * uniforms.terrain.x;
  let spacing = uniforms.terrain.x / f32(dimensions.x - 1u);
  let height = terrainHeight(p, uniforms.environment.x);
  let left = terrainHeight(p - vec2<f32>(spacing, 0.0), uniforms.environment.x);
  let right = terrainHeight(p + vec2<f32>(spacing, 0.0), uniforms.environment.x);
  let back = terrainHeight(p - vec2<f32>(0.0, spacing), uniforms.environment.x);
  let front = terrainHeight(p + vec2<f32>(0.0, spacing), uniforms.environment.x);
  let normal = normalize(vec3<f32>(left - right, spacing * 2.0, back - front));
  textureStore(field_out, vec2<i32>(id.xy), vec4<f32>(height, normal.x, normal.z, 0.0));
}

// ---------------------------------------------------------------------------
// Water simulation compute
// ---------------------------------------------------------------------------
struct SimulationParams {
  impulse: vec4<f32>,
  stepFoamShift: vec4<f32>,
}
var<uniform> sim_params: SimulationParams;
var previous_state: texture_2d<f32>;
var next_state: texture_storage_2d<rgba16float, write>;
var terrain_field: texture_2d<f32>;
var long_field0: texture_2d<f32>;
var long_field1: texture_2d<f32>;
var medium_field0: texture_2d<f32>;
var medium_field1: texture_2d<f32>;
var spectrum_sampler: sampler;

const GRAVITY = 9.81;
const MIN_DEPTH = 0.035;

struct CellState {
  eta: f32,
  q: vec2<f32>,
  foam: f32,
  bottom: f32,
  depth: f32,
}

fn clampedCoord(coord: vec2<i32>, dimensions: vec2<u32>) -> vec2<i32> {
  return clamp(coord, vec2<i32>(0), vec2<i32>(dimensions) - vec2<i32>(1));
}

fn worldPosition(coord: vec2<i32>, dimensions: vec2<u32>) -> vec2<f32> {
  let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dimensions);
  return uniforms.simulation.xy + (uv - vec2<f32>(0.5)) * uniforms.simulation.z;
}

fn terrainAtWorld(p: vec2<f32>) -> f32 {
  let terrainDimensions = textureDimensions(terrain_field);
  let uv = clamp(p / uniforms.terrain.x + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
  let coord = vec2<i32>(round(uv * vec2<f32>(terrainDimensions - vec2<u32>(1))));
  return textureLoad(terrain_field, coord, 0).r;
}

fn spectralBoundaryState(p: vec2<f32>, depth: f32) -> vec4<f32> {
  let longUv = fract(p / 240.0 + vec2<f32>(0.5));
  let mediumUv = fract(p / 64.0 + vec2<f32>(0.5));
  let long0 = textureSampleLevel(long_field0, spectrum_sampler, longUv, 0.0);
  let long1 = textureSampleLevel(long_field1, spectrum_sampler, longUv, 0.0);
  let medium0 = textureSampleLevel(medium_field0, spectrum_sampler, mediumUv, 0.0);
  let medium1 = textureSampleLevel(medium_field1, spectrum_sampler, mediumUv, 0.0);
  let longHeight = long0.b;
  let mediumHeight = medium0.b;
  let eta = longHeight + mediumHeight
    + 0.14 * (longHeight * longHeight - 0.080)
    + 0.32 * (mediumHeight * mediumHeight - 0.030);
  // The boundary transport follows the dominant spectrum direction. Interior
  // momentum immediately becomes bathymetry-aware through the conservative
  // flux. This is a relaxation boundary, not a second rendered wave layer.
  let meanDirection = normalize(vec2<f32>(0.887, -0.462));
  let direction = normalize(meanDirection - (long1.rg + medium1.rg) * 0.055);
  let phaseSpeed = sqrt(GRAVITY * max(depth, MIN_DEPTH));
  let crossDerivative = long0.a * 1.18 + medium0.a * 1.05;
  let horizontalDerivative = long1.ba * 1.18 + medium1.ba * 1.05;
  let jacobian = (1.0 + horizontalDerivative.x) * (1.0 + horizontalDerivative.y) - crossDerivative * crossDerivative;
  return vec4<f32>(eta, direction * eta * phaseSpeed, max(0.0, 1.0 - jacobian));
}

fn loadCell(coordIn: vec2<i32>, dimensions: vec2<u32>) -> CellState {
  let coord = clampedCoord(coordIn, dimensions);
  let raw = textureLoad(previous_state, coord, 0);
  let bottom = terrainAtWorld(worldPosition(coord, dimensions));
  let depth = max(uniforms.sunWater.w + raw.r - bottom, 0.0);
  var result: CellState;
  result.eta = raw.r;
  result.q = select(raw.gb, vec2<f32>(0.0), depth <= MIN_DEPTH);
  result.foam = raw.a;
  result.bottom = bottom;
  result.depth = depth;
  return result;
}

fn conservativeState(cell: CellState, reconstructedDepth: f32) -> vec3<f32> {
  let scale = select(reconstructedDepth / max(cell.depth, MIN_DEPTH), 0.0, cell.depth <= MIN_DEPTH);
  return vec3<f32>(reconstructedDepth, cell.q * scale);
}

fn physicalFluxX(state: vec3<f32>) -> vec3<f32> {
  let h = max(state.x, MIN_DEPTH);
  let velocity = state.yz / h;
  return vec3<f32>(state.y, state.y * velocity.x + 0.5 * GRAVITY * state.x * state.x, state.y * velocity.y);
}

fn physicalFluxY(state: vec3<f32>) -> vec3<f32> {
  let h = max(state.x, MIN_DEPTH);
  let velocity = state.yz / h;
  return vec3<f32>(state.z, state.z * velocity.x, state.z * velocity.y + 0.5 * GRAVITY * state.x * state.x);
}

fn hydrostaticPair(a: CellState, b: CellState) -> array<vec3<f32>, 2> {
  let interfaceBottom = max(a.bottom, b.bottom);
  let surfaceA = uniforms.sunWater.w + a.eta;
  let surfaceB = uniforms.sunWater.w + b.eta;
  let hA = max(0.0, surfaceA - interfaceBottom);
  let hB = max(0.0, surfaceB - interfaceBottom);
  return array<vec3<f32>, 2>(conservativeState(a, hA), conservativeState(b, hB));
}

fn rusanovX(a: CellState, b: CellState) -> vec3<f32> {
  let pair = hydrostaticPair(a, b);
  let left = pair[0];
  let right = pair[1];
  let uLeft = select(left.y / max(left.x, MIN_DEPTH), 0.0, left.x <= MIN_DEPTH);
  let uRight = select(right.y / max(right.x, MIN_DEPTH), 0.0, right.x <= MIN_DEPTH);
  let speed = max(abs(uLeft) + sqrt(GRAVITY * left.x), abs(uRight) + sqrt(GRAVITY * right.x));
  return 0.5 * (physicalFluxX(left) + physicalFluxX(right)) - 0.5 * speed * (right - left);
}

fn rusanovY(a: CellState, b: CellState) -> vec3<f32> {
  let pair = hydrostaticPair(a, b);
  let south = pair[0];
  let north = pair[1];
  let vSouth = select(south.z / max(south.x, MIN_DEPTH), 0.0, south.x <= MIN_DEPTH);
  let vNorth = select(north.z / max(north.x, MIN_DEPTH), 0.0, north.x <= MIN_DEPTH);
  let speed = max(abs(vSouth) + sqrt(GRAVITY * south.x), abs(vNorth) + sqrt(GRAVITY * north.x));
  return 0.5 * (physicalFluxY(south) + physicalFluxY(north)) - 0.5 * speed * (north - south);
}

fn sidePressureCorrection(originalDepth: f32, reconstructedDepth: f32) -> f32 {
  return 0.5 * GRAVITY * (originalDepth * originalDepth - reconstructedDepth * reconstructedDepth);
}

@compute @workgroup_size(16, 16)
fn simulate(@builtin(global_invocation_id) id: vec3<u32>) {
  let dimensions = textureDimensions(next_state);
  if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }
  let coord = vec2<i32>(id.xy);
  let center = loadCell(coord, dimensions);
  let west = loadCell(coord - vec2<i32>(1, 0), dimensions);
  let east = loadCell(coord + vec2<i32>(1, 0), dimensions);
  let south = loadCell(coord - vec2<i32>(0, 1), dimensions);
  let north = loadCell(coord + vec2<i32>(0, 1), dimensions);
  let cellSize = uniforms.simulation.z / f32(dimensions.x);
  let dt = sim_params.stepFoamShift.x;

  let eastPair = hydrostaticPair(center, east);
  let westPair = hydrostaticPair(west, center);
  let northPair = hydrostaticPair(center, north);
  let southPair = hydrostaticPair(south, center);
  var eastFlux = rusanovX(center, east);
  var westFlux = rusanovX(west, center);
  var northFlux = rusanovY(center, north);
  var southFlux = rusanovY(south, center);
  eastFlux.y += sidePressureCorrection(center.depth, eastPair[0].x);
  westFlux.y += sidePressureCorrection(center.depth, westPair[1].x);
  northFlux.z += sidePressureCorrection(center.depth, northPair[0].x);
  southFlux.z += sidePressureCorrection(center.depth, southPair[1].x);

  var next = vec3<f32>(center.depth, center.q) - dt * ((eastFlux - westFlux) + (northFlux - southFlux)) / cellSize;
  next.x = max(next.x, 0.0);
  var nextDepth = next.x;
  var nextQ = select(next.yz, vec2<f32>(0.0), nextDepth <= MIN_DEPTH);
  let speed = length(nextQ) / max(nextDepth, MIN_DEPTH);
  let manning = 0.018;
  let friction = GRAVITY * manning * manning * speed / max(pow(max(nextDepth, MIN_DEPTH), 1.333333), 0.001);
  nextQ /= 1.0 + dt * friction;

  let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(dimensions);
  let radius = max(sim_params.impulse.w, 0.0001);
  let impulseDistance = length((uv - sim_params.impulse.xy) / radius);
  let impulse = exp(-impulseDistance * impulseDistance * 3.2);
  let ring = exp(-pow(impulseDistance - 0.72, 2.0) * 18.0);
  nextDepth = max(0.0, nextDepth + (impulse - ring * 0.28) * sim_params.impulse.z);
  let impulseDirection = normalize(vec2<f32>(uniforms.player.z, uniforms.player.w) + vec2<f32>(0.0001, 0.0));
  nextQ += impulseDirection * ring * sim_params.impulse.z * 1.6;

  // Couple the far-field FFT to the nonlinear domain. A strong sponge forces
  // the outer band to the incident sea state, while deeper interior cells get
  // a much weaker source during warm-up. Shallow cells are then owned by the
  // conservative solver, allowing bathymetric refraction and run-up.
  let edgeDistance = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
  let sponge = 1.0 - smoothstep(0.0, 0.085, edgeDistance);
  let stillDepth = max(uniforms.sunWater.w - center.bottom, 0.0);
  let boundary = spectralBoundaryState(worldPosition(coord, dimensions), stillDepth);
  let deepWarmup = smoothstep(4.8, 9.5, stillDepth) * (1.0 - sponge) * 0.42;
  let coupling = min(1.0, dt * (12.0 * sponge + 2.4 * deepWarmup));
  nextDepth = mix(nextDepth, max(stillDepth + boundary.x, 0.0), coupling);
  // Linear shallow-water transport is q = c * eta. Multiplying by depth a
  // second time over-forces the wet/dry front and produces a vertical wall.
  nextQ = mix(nextQ, boundary.yz, coupling);

  let velocity = nextQ / max(nextDepth, MIN_DEPTH);
  let backtraceUv = clamp(uv - velocity * dt / uniforms.simulation.z, vec2<f32>(0.002), vec2<f32>(0.998));
  let backtracedFoam = textureSampleLevel(previous_state, spectrum_sampler, backtraceUv, 0.0).a;
  let neighbourFoam = (west.foam + east.foam + south.foam + north.foam) * 0.25;
  var foam = mix(backtracedFoam, neighbourFoam, min(0.11, dt * 1.4));
  let froude = speed / max(sqrt(GRAVITY * nextDepth), 0.001);
  let surfaceCompression = max(0.0, -(east.q.x - west.q.x + north.q.y - south.q.y) / (2.0 * cellSize));
  let breakingBirth = smoothstep(0.58, 0.92, froude) * smoothstep(0.03, 0.32, surfaceCompression);
  let shorelineBirth = (1.0 - smoothstep(0.16, 1.7, nextDepth)) * smoothstep(0.03, 0.24, speed);
  let spectralBirth = smoothstep(0.115, 0.31, boundary.w) * smoothstep(0.27, 0.76, boundary.x);
  let shorelineWaveBirth = (1.0 - smoothstep(0.10, 1.55, nextDepth)) * smoothstep(0.18, 0.64, boundary.x);
  foam *= exp(-dt * 0.58);
  foam += dt * (spectralBirth * 0.48 + breakingBirth * 2.4 + shorelineBirth * 0.52 + shorelineWaveBirth * 1.25) * sim_params.stepFoamShift.y;
  foam = max(foam, ring * abs(sim_params.impulse.z) * 4.0 * sim_params.stepFoamShift.y);

  let eta = nextDepth + center.bottom - uniforms.sunWater.w;
  textureStore(next_state, coord, vec4<f32>(clamp(eta, -1.8, 1.8), clamp(nextQ, vec2<f32>(-12.0), vec2<f32>(12.0)), clamp(foam, 0.0, 1.0)));
}

// ---------------------------------------------------------------------------
// Breaker event compute
// ---------------------------------------------------------------------------
var previous_events: texture_2d<f32>;
var next_events: texture_storage_2d<rgba16float, write>;
var water_state: texture_2d<f32>;
var field_sampler: sampler;

fn frontPosition(time: f32) -> f32 {
  let travellingPhase = time * 2.4 + 12.0;
  return travellingPhase - floor(travellingPhase / 72.0) * 72.0 - 36.0;
}

fn eventHistory(coord: i32) -> f32 {
  return textureLoad(previous_events, vec2<i32>(clamp(coord, 0, 255), 0), 0).r;
}

@compute @workgroup_size(64, 1)
fn updateBreakerEvents(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= 256u) { return; }
  let uv = (f32(id.x) + 0.5) / 256.0;
  let along = mix(-180.0, 180.0, uv);
  let travelDirection = normalize(vec2<f32>(0.887, -0.462));
  let tangentDirection = vec2<f32>(-travelDirection.y, travelDirection.x);
  let time = uniforms.cameraTime.w;
  let meander = sin(along * 0.055 + time * 0.055 + 0.7) * 3.8
    + sin(along * 0.14 - time * 0.032 - 1.3) * 1.2;
  let p = tangentDirection * along + travelDirection * (frontPosition(time) + meander);

  let terrainUv = clamp(p / uniforms.terrain.x + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
  let bottom = textureSampleLevel(terrain_field, field_sampler, terrainUv, 0.0).r;
  let stillDepth = max(uniforms.sunWater.w - bottom, 0.0);
  let simulationUv = (p - uniforms.simulation.xy) / uniforms.simulation.z + vec2<f32>(0.5);
  let simulationInside = step(0.0, simulationUv.x) * step(0.0, simulationUv.y) * step(simulationUv.x, 1.0) * step(simulationUv.y, 1.0);
  let state = textureSampleLevel(water_state, field_sampler, clamp(simulationUv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0) * simulationInside;
  let dynamicDepth = max(stillDepth + state.r, 0.035);
  let speed = length(state.gb) / dynamicDepth;
  let froude = speed / max(sqrt(9.81 * dynamicDepth), 0.001);

  let longUv = fract(p / 240.0 + vec2<f32>(0.5));
  let mediumUv = fract(p / 64.0 + vec2<f32>(0.5));
  let long0 = textureSampleLevel(long_field0, spectrum_sampler, longUv, 0.0);
  let long1 = textureSampleLevel(long_field1, spectrum_sampler, longUv, 0.0);
  let medium0 = textureSampleLevel(medium_field0, spectrum_sampler, mediumUv, 0.0);
  let medium1 = textureSampleLevel(medium_field1, spectrum_sampler, mediumUv, 0.0);
  let crossDerivative = long0.a * 1.18 + medium0.a * 1.05;
  let horizontalDerivative = long1.ba * 1.18 + medium1.ba * 1.05;
  let jacobian = (1.0 + horizontalDerivative.x) * (1.0 + horizontalDerivative.y) - crossDerivative * crossDerivative;
  let compression = max(0.0, 1.0 - jacobian);
  let slope = length(long1.rg + medium1.rg);
  let spectralInstability = smoothstep(0.035, 0.160, compression)
    * mix(0.34, 1.0, smoothstep(0.045, 0.205, slope));
  let depthRatio = abs(state.r) / max(dynamicDepth, 0.12);
  let nearshoreInstability = (1.0 - smoothstep(2.2, 6.0, dynamicDepth))
    * max(smoothstep(0.46, 0.86, froude), smoothstep(0.38, 0.76, depthRatio));
  let targetInstability = clamp(max(spectralInstability, nearshoreInstability), 0.0, 1.0);

  // Lateral history diffusion gives contiguous breaking segments. Fast attack
  // and slow release implement the persistent breaking state used by practical
  // nearshore solvers instead of flickering on a single threshold crossing.
  let center = eventHistory(i32(id.x));
  let history = center * 0.50 + eventHistory(i32(id.x) - 1) * 0.25 + eventHistory(i32(id.x) + 1) * 0.25;
  let rate = select(0.62, 7.5, targetInstability > history);
  let blend = 1.0 - exp(-rate / 60.0);
  let activation = mix(history, targetInstability, blend);
  textureStore(next_events, vec2<i32>(i32(id.x), 0), vec4<f32>(activation, spectralInstability, nearshoreInstability, compression));
}

// ---------------------------------------------------------------------------
// Spectrum evolution compute
// ---------------------------------------------------------------------------
var initial_spectrum: texture_2d<f32>;
var wave_data: texture_2d<f32>;
var field0: texture_storage_2d<rgba16float, write>;
var field1: texture_storage_2d<rgba16float, write>;

fn complexMultiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(8, 8)
fn evolveSpectrum(@builtin(global_invocation_id) id: vec3<u32>) {
  let dimensions = textureDimensions(field0);
  if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }
  let coord = vec2<i32>(id.xy);
  let initial = textureLoad(initial_spectrum, coord, 0);
  let wave = textureLoad(wave_data, coord, 0);
  let phase = wave.w * uniforms.cameraTime.w;
  let exponent = vec2<f32>(cos(phase), sin(phase));
  let h = complexMultiply(initial.xy, exponent) + complexMultiply(initial.zw, vec2<f32>(exponent.x, -exponent.y));
  let ih = vec2<f32>(-h.y, h.x);
  let displacementX = ih * wave.x * wave.y;
  let displacementY = h;
  let displacementZ = ih * wave.z * wave.y;
  let displacementXdx = -h * wave.x * wave.x * wave.y;
  let displacementYdx = ih * wave.x;
  let displacementZdx = -h * wave.x * wave.z * wave.y;
  let displacementYdz = ih * wave.z;
  let displacementZdz = -h * wave.z * wave.z * wave.y;
  let dxDz = vec2<f32>(displacementX.x - displacementZ.y, displacementX.y + displacementZ.x);
  let dyDxz = vec2<f32>(displacementY.x - displacementZdx.y, displacementY.y + displacementZdx.x);
  let dyxDyz = vec2<f32>(displacementYdx.x - displacementYdz.y, displacementYdx.y + displacementYdz.x);
  let dxxDzz = vec2<f32>(displacementXdx.x - displacementZdz.y, displacementXdx.y + displacementZdz.x);
  textureStore(field0, coord, vec4<f32>(dxDz, dyDxz));
  textureStore(field1, coord, vec4<f32>(dyxDyz, dxxDzz));
}

// ---------------------------------------------------------------------------
// Spectral inverse FFT compute
// ---------------------------------------------------------------------------
struct FftParams {
  axis: u32,
  stage: u32,
  size: u32,
  finalize: u32,
}
var<uniform> fft_params: FftParams;
var twiddle_table: texture_2d<f32>;
var input0: texture_2d<f32>;
var input1: texture_2d<f32>;
var output0: texture_storage_2d<rgba16float, write>;
var output1: texture_storage_2d<rgba16float, write>;

fn butterfly(a: vec4<f32>, b: vec4<f32>, twiddle: vec2<f32>) -> vec4<f32> {
  return vec4<f32>(a.xy + complexMultiply(twiddle, b.xy), a.zw + complexMultiply(twiddle, b.zw));
}

@compute @workgroup_size(8, 8)
fn inverseFftStage(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= fft_params.size || id.y >= fft_params.size) { return; }
  let outputCoord = vec2<i32>(id.xy);
  let transformIndex = select(id.y, id.x, fft_params.axis == 0u);
  let data = textureLoad(twiddle_table, vec2<i32>(i32(transformIndex), i32(fft_params.stage)), 0);
  let first = i32(round(data.z));
  let second = i32(round(data.w));
  var coord0 = vec2<i32>(i32(id.x), i32(id.y));
  var coord1 = coord0;
  if (fft_params.axis == 0u) {
    coord0.x = first;
    coord1.x = second;
  } else {
    coord0.y = first;
    coord1.y = second;
  }
  let inverseTwiddle = vec2<f32>(data.x, -data.y);
  var value0 = butterfly(textureLoad(input0, coord0, 0), textureLoad(input0, coord1, 0), inverseTwiddle);
  var value1 = butterfly(textureLoad(input1, coord0, 0), textureLoad(input1, coord1, 0), inverseTwiddle);
  if (fft_params.finalize == 1u) {
    let checker = 1.0 - 2.0 * f32((id.x + id.y) % 2u);
    value0 *= checker;
    value1 *= checker;
  }
  textureStore(output0, outputCoord, value0);
  textureStore(output1, outputCoord, value1);
}

// ---------------------------------------------------------------------------
// Color functions
// ---------------------------------------------------------------------------
fn linearToSrgb(value: vec3<f32>) -> vec3<f32> {
  return mix(value * 12.92, 1.055 * pow(max(value, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055, step(vec3<f32>(0.0031308), value));
}

fn aces(color: vec3<f32>) -> vec3<f32> {
  return clamp((color * (2.51 * color + vec3<f32>(0.03))) / (color * (2.43 * color + vec3<f32>(0.59)) + vec3<f32>(0.14)), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn cloudHash3(pInput: vec3<f32>) -> f32 {
  var p = fract(pInput * 0.1031);
  p += vec3<f32>(dot(p, p.yzx + vec3<f32>(33.33)));
  return fract((p.x + p.y) * p.z);
}

fn cloudNoise3(p: vec3<f32>) -> f32 {
  let cell = floor(p);
  var local = fract(p);
  local = local * local * (vec3<f32>(3.0) - 2.0 * local);
  let n000 = cloudHash3(cell + vec3<f32>(0.0, 0.0, 0.0));
  let n100 = cloudHash3(cell + vec3<f32>(1.0, 0.0, 0.0));
  let n010 = cloudHash3(cell + vec3<f32>(0.0, 1.0, 0.0));
  let n110 = cloudHash3(cell + vec3<f32>(1.0, 1.0, 0.0));
  let n001 = cloudHash3(cell + vec3<f32>(0.0, 0.0, 1.0));
  let n101 = cloudHash3(cell + vec3<f32>(1.0, 0.0, 1.0));
  let n011 = cloudHash3(cell + vec3<f32>(0.0, 1.0, 1.0));
  let n111 = cloudHash3(cell + vec3<f32>(1.0, 1.0, 1.0));
  let nearPlane = mix(mix(n000, n100, local.x), mix(n010, n110, local.x), local.y);
  let farPlane = mix(mix(n001, n101, local.x), mix(n011, n111, local.x), local.y);
  return mix(nearPlane, farPlane, local.z);
}

fn skyColor(direction: vec3<f32>, time: f32, sunDirection: vec3<f32>) -> vec3<f32> {
  let elevation = direction.y;
  let upper = smoothstep(-0.035, 0.34, elevation);
  // Lower-energy linear-light values leave headroom for the sun and clouds.
  // The old near-white dome flattened both the sky and its water reflection.
  var color = mix(vec3<f32>(0.34, 0.54, 0.64), vec3<f32>(0.070, 0.26, 0.43), upper);
  color = mix(vec3<f32>(0.055, 0.22, 0.31), color, smoothstep(-0.34, 0.055, elevation));
  let sunDot = max(dot(direction, sunDirection), 0.0);
  let horizon = exp(-abs(elevation) * 12.0);
  color = mix(color, vec3<f32>(0.72, 0.55, 0.31), pow(sunDot, 4.0) * horizon * 0.12);
  color += vec3<f32>(0.28, 0.42, 0.62) * pow(sunDot, 20.0) * 0.08;
  let drift = vec3<f32>(time * 0.0040, -time * 0.0014, time * 0.0023);
  let cloudPoint = direction * 10.5 + drift;
  let cloudField = cloudNoise3(cloudPoint) * 0.54
    + cloudNoise3(cloudPoint * 2.03 + vec3<f32>(7.1, -3.4, 5.8)) * 0.29
    + cloudNoise3(cloudPoint * 4.07 + vec3<f32>(-2.7, 9.3, 1.9)) * 0.17;
  let envelope = smoothstep(-0.045, 0.018, elevation) * (1.0 - smoothstep(0.30, 0.43, elevation));
  let cloudBody = smoothstep(0.535, 0.68, cloudField) * envelope;
  let cloudCore = smoothstep(0.63, 0.78, cloudField) * envelope;
  let cloudEdge = (smoothstep(0.51, 0.59, cloudField) - smoothstep(0.67, 0.76, cloudField)) * envelope;
  let cloudShade = mix(vec3<f32>(0.42, 0.50, 0.55), vec3<f32>(0.82, 0.76, 0.63), pow(sunDot, 0.35));
  color = mix(color, cloudShade, cloudBody * 0.42);
  color -= vec3<f32>(0.08, 0.10, 0.12) * cloudCore * (1.0 - sunDot) * 0.28;
  color += vec3<f32>(0.60, 0.49, 0.30) * cloudEdge * sunDot * 0.055;
  return color;
}

// ---------------------------------------------------------------------------
// Sky
// ---------------------------------------------------------------------------
struct SkyOutput { @builtin(position) position: vec4<f32>, @location(0) ndc: vec2<f32> }

@vertex fn skyVertex(@builtin(vertex_index) id: u32) -> SkyOutput {
  var positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var output: SkyOutput;
  output.position = vec4<f32>(positions[id], 0.999999, 1.0);
  output.ndc = positions[id];
  return output;
}

@fragment fn skyFragment(input: SkyOutput) -> @location(0) vec4<f32> {
  let ray = normalize(uniforms.cameraForward.xyz + input.ndc.x * uniforms.cameraRight.xyz * uniforms.cameraRight.w + input.ndc.y * uniforms.cameraUp.xyz * uniforms.cameraUp.w);
  var color = skyColor(ray, uniforms.cameraTime.w, normalize(uniforms.sunWater.xyz));
  if (uniforms.terrain.w > 0.5) {
    let upward = smoothstep(-0.42, 0.72, ray.y);
    let volume = mix(vec3<f32>(0.012, 0.155, 0.158), vec3<f32>(0.050, 0.385, 0.335), upward);
    let lightColumn = pow(max(dot(ray, normalize(uniforms.sunWater.xyz)), 0.0), 14.0) * upward;
    color = volume + vec3<f32>(0.18, 0.30, 0.23) * lightColumn * 0.22;
  }
  return vec4<f32>(linearToSrgb(aces(color)), 1.0);
}

// ---------------------------------------------------------------------------
// Terrain render
// ---------------------------------------------------------------------------
struct TerrainOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) fieldUv: vec2<f32>,
}

@vertex fn terrainVertex(@builtin(vertex_index) vertexId: u32) -> TerrainOutput {
  var corners = array<vec2<u32>, 6>(
    vec2<u32>(0u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u, 0u),
    vec2<u32>(0u, 1u), vec2<u32>(1u, 1u), vec2<u32>(1u, 0u)
  );
  let resolution = u32(uniforms.environment.y);
  let cellId = vertexId / 6u;
  let cell = vec2<u32>(cellId % resolution, cellId / resolution);
  let grid = cell + corners[vertexId % 6u];
  let uv = vec2<f32>(grid) / f32(resolution);
  let sample = textureSampleLevel(terrain_field, field_sampler, uv, 0.0);
  let world = vec3<f32>((uv.x - 0.5) * uniforms.terrain.x, sample.r, (uv.y - 0.5) * uniforms.terrain.x);
  var output: TerrainOutput;
  output.position = uniforms.viewProj * vec4<f32>(world, 1.0);
  output.world = world;
  output.fieldUv = uv;
  return output;
}

@fragment fn terrainFragment(input: TerrainOutput) -> @location(0) vec4<f32> {
  let field = textureSample(terrain_field, field_sampler, input.fieldUv);
  let normalY = sqrt(max(1.0 - field.g * field.g - field.b * field.b, 0.0001));
  let N = normalize(vec3<f32>(field.g, normalY, field.b));
  let L = normalize(uniforms.sunWater.xyz);
  let diffuse = clamp(dot(N, L) * 0.56 + 0.48, 0.0, 1.0);
  let p = input.world.xz;
  let broad = valueNoise(p * 0.075 - vec2<f32>(8.1, -2.4));
  let grain = valueNoise(p * 0.38 + vec2<f32>(4.7, -9.2));
  let geology = valueNoise(p * 0.021 + vec2<f32>(13.2, -6.7));
  let erosion = valueNoise(vec2<f32>(p.x * 0.055 + p.y * 0.018, p.y * 0.19 - p.x * 0.025));
  let sedimentMacro = valueNoise(p * 0.018 + vec2<f32>(-11.4, 6.8));
  let sedimentMeso = valueNoise(vec2<f32>(p.x * 0.092 + p.y * 0.027, p.y * 0.105 - p.x * 0.021) + vec2<f32>(3.9, -7.1));
  let simulationUv = (p - uniforms.simulation.xy) / uniforms.simulation.z + vec2<f32>(0.5);
  let inSimulation = step(0.0, simulationUv.x) * step(simulationUv.x, 1.0)
    * step(0.0, simulationUv.y) * step(simulationUv.y, 1.0);
  let shoreState = textureSample(water_state, field_sampler, clamp(simulationUv, vec2<f32>(0.0), vec2<f32>(1.0)));
  let localWaterLevel = uniforms.sunWater.w
    + clamp(shoreState.r, -0.16, 0.18) * inSimulation * uniforms.environment.x;
  let sandSource = mix(vec3<f32>(0.22, 0.185, 0.115), vec3<f32>(0.43, 0.345, 0.19), sedimentMacro)
    * (0.88 + broad * 0.15 + grain * 0.045 + (sedimentMeso - 0.5) * 0.18);
  let granularVariation = (broad - 0.5) * 0.07 + (grain - 0.5) * 0.025;
  var color = mix(sandSource * vec3<f32>(0.74, 0.84, 0.72), vec3<f32>(0.045, 0.17, 0.145), 0.10)
    * mix(0.68, 1.08, diffuse) * (1.0 + granularVariation);
  let depth = max(0.0, localWaterLevel - input.world.y);
  let exposed = smoothstep(localWaterLevel + 0.18, localWaterLevel + 0.64, input.world.y) * uniforms.environment.x;
  let sandBase = mix(vec3<f32>(0.235, 0.135, 0.050), vec3<f32>(0.48, 0.315, 0.115), broad);
  let rockBase = mix(vec3<f32>(0.22, 0.175, 0.125), vec3<f32>(0.39, 0.285, 0.175), geology);
  let rockMask = smoothstep(0.18, 0.58, 1.0 - N.y) * 0.72 + smoothstep(0.68, 0.90, erosion) * 0.24;
  let ripplePhase = p.x * 0.21 + p.y * 0.065 + valueNoise(p * 0.030 + vec2<f32>(2.7, -5.4)) * 5.2;
  let sandRipple = sin(ripplePhase) * 0.5 + 0.5;
  let duneMacro = valueNoise(p * 0.026 + vec2<f32>(-3.4, 7.1));
  let duneMeso = valueNoise(vec2<f32>(p.x * 0.063 + p.y * 0.017, p.y * 0.071 - p.x * 0.012) + vec2<f32>(6.2, -1.8));
  let duneTone = (duneMacro - 0.5) * 0.31
    + (duneMeso - 0.5) * 0.15
    + (sandRipple - 0.5) * 0.045;
  let sunwardCrest = clamp(dot(N.xz, normalize(vec2<f32>(-0.52, -0.80))) * 0.5 + 0.5, 0.0, 1.0);
  let windPolish = smoothstep(0.64, 0.90, N.y) * smoothstep(0.48, 0.76, duneMeso);
  let sandPalette = mix(sandBase * vec3<f32>(0.84, 0.89, 0.82), sandBase * vec3<f32>(1.13, 1.07, 0.91), duneMacro);
  let elevationTone = smoothstep(localWaterLevel + 0.30, localWaterLevel + 4.8, input.world.y);
  let elevationSand = mix(vec3<f32>(0.255, 0.145, 0.052), vec3<f32>(0.53, 0.35, 0.135), elevationTone);
  let drySand = mix(sandBase, rockBase, clamp(rockMask, 0.0, 0.82))
    * mix(0.72, 1.03, diffuse)
    * (0.84 + broad * 0.15 + grain * 0.047 + (erosion - 0.5) * 0.065 + duneTone);
  let polishedSand = mix(sandPalette, elevationSand, 0.42) * mix(0.82, 1.04, diffuse);
  let drySandLayered = mix(drySand, polishedSand, 0.28 + windPolish * 0.24)
    * mix(0.83, 1.07, sunwardCrest);
  let coast = smoothstep(localWaterLevel - 0.30, localWaterLevel + 0.34, input.world.y) * uniforms.environment.x;
  let solverWash = inSimulation * uniforms.environment.x
    * smoothstep(0.018, 0.22, shoreState.a)
    * (1.0 - smoothstep(0.05, 0.62, abs(input.world.y - localWaterLevel)));
  let wetSand = mix(vec3<f32>(0.18, 0.135, 0.088), vec3<f32>(0.255, 0.185, 0.105), broad)
    * mix(0.74, 0.98, diffuse) * (0.91 + grain * 0.055 + sandRipple * 0.025);
  color = mix(color, wetSand, coast);
  color = mix(color, wetSand * vec3<f32>(0.82, 0.88, 0.84), solverWash * 0.12);
  color = mix(color, drySandLayered, exposed);
  // Project the seabed point toward the refracted sun ray before sampling the
  // actual animated surface derivatives. This keeps the caustic tied to the
  // spectral water instead of painting a cellular texture onto the sand.
  let refractedSunOffset = L.xz / max(L.y, 0.12) * depth * 0.18;
  let surfaceP = p - refractedSunOffset;
  let mediumUv = fract(surfaceP / 64.0 + vec2<f32>(0.5));
  let shortUv = fract(surfaceP / 12.0 + vec2<f32>(0.5));
  let medium0 = textureSample(medium_field0, spectrum_sampler, mediumUv);
  let medium1 = textureSample(medium_field1, spectrum_sampler, mediumUv);
  let short0 = textureSample(short_field0, spectrum_sampler, shortUv);
  let short1 = textureSample(short_field1, spectrum_sampler, shortUv);
  let mediumCross = medium0.a * 1.05;
  let mediumDerivative = medium1.ba * 1.05;
  let shortCross = short0.a * 0.40;
  let shortDerivative = short1.ba * 0.40;
  let mediumJacobian = (1.0 + mediumDerivative.x) * (1.0 + mediumDerivative.y) - mediumCross * mediumCross;
  let shortJacobian = (1.0 + shortDerivative.x) * (1.0 + shortDerivative.y) - shortCross * shortCross;
  let surfaceFocus = max(0.0, 1.0 - mediumJacobian) * 0.48 + max(0.0, 1.0 - shortJacobian) * 0.52;
  let focusedLight = pow(smoothstep(0.060, 0.27, surfaceFocus), 2.0)
    * smoothstep(0.6, 2.2, depth) * (1.0 - smoothstep(11.0, 20.0, depth));
  color *= 0.94 + focusedLight * 0.14 * (1.0 - exposed);
  color += vec3<f32>(0.095, 0.105, 0.045) * focusedLight * 0.060 * (1.0 - exposed);
  let distanceToEye = distance(uniforms.cameraTime.xyz, input.world);
  let underwater = uniforms.terrain.w > 0.5;
  let density = select(0.00155, 0.0075, underwater);
  var fog = 1.0 - exp(-max(distanceToEye - select(20.0, 2.0, underwater), 0.0) * density);
  let dryLand = exposed * (1.0 - select(0.0, 1.0, underwater));
  let oceanRadialFog = smoothstep(116.0, 145.0, length(p));
  let islandRadialFog = smoothstep(166.0, 205.0, length(p)) * 0.58;
  fog = clamp(fog + mix(oceanRadialFog, islandRadialFog, dryLand), 0.0, mix(0.99, 0.72, dryLand));
  let waterAerial = select(vec3<f32>(0.42, 0.66, 0.71), vec3<f32>(0.012, 0.205, 0.185), underwater);
  let aerial = mix(waterAerial, vec3<f32>(0.24, 0.39, 0.43), dryLand);
  color = mix(color, aerial, fog);
  return vec4<f32>(linearToSrgb(aces(color)), 1.0);
}

// ---------------------------------------------------------------------------
// Water render
// ---------------------------------------------------------------------------
var short_field0: texture_2d<f32>;
var short_field1: texture_2d<f32>;
var breaker_events: texture_2d<f32>;

struct WaterOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) fieldUv: vec2<f32>,
  @location(3) simulationUv: vec2<f32>,
  @location(4) waveHeight: f32,
  @location(5) compression: f32,
  @location(6) breakerLip: f32,
  @location(7) breakerCoord: vec2<f32>,
  @location(8) surfaceKind: f32,
}

struct SurfaceEvaluation {
  world: vec3<f32>,
  tangentPX: vec3<f32>,
  tangentPZ: vec3<f32>,
  fieldUv: vec2<f32>,
  simulationUv: vec2<f32>,
  waveHeight: f32,
  compression: f32,
  breakerLip: f32,
}

fn simulationSample(uv: vec2<f32>) -> vec4<f32> {
  let inside = step(0.0, uv.x) * step(0.0, uv.y) * step(uv.x, 1.0) * step(uv.y, 1.0);
  return textureSampleLevel(water_state, field_sampler, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0) * inside;
}

fn dielectricFresnel(cosine: f32) -> f32 {
  let eta = 1.0 / 1.333;
  let sinTransmittedSquared = eta * eta * max(0.0, 1.0 - cosine * cosine);
  if (sinTransmittedSquared >= 1.0) { return 1.0; }
  let transmittedCosine = sqrt(max(0.0, 1.0 - sinTransmittedSquared));
  let parallel = (cosine - 1.333 * transmittedCosine) / max(cosine + 1.333 * transmittedCosine, 0.0001);
  let perpendicular = (transmittedCosine - 1.333 * cosine) / max(transmittedCosine + 1.333 * cosine, 0.0001);
  return 0.5 * (parallel * parallel + perpendicular * perpendicular);
}

fn smithVisibility(cosine: f32, meanSquareSlope: f32) -> f32 {
  let tangentSquared = max(0.0, 1.0 - cosine * cosine) / max(cosine * cosine, 0.0001);
  return 2.0 / (1.0 + sqrt(1.0 + meanSquareSlope * tangentSquared));
}

fn oceanSunGlitter(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>) -> f32 {
  let ndv = max(dot(N, V), 0.001);
  let ndl = max(dot(N, L), 0.001);
  let H = normalize(V + L);
  let ndh = max(dot(N, H), 0.001);
  let windWorld = normalize(vec3<f32>(0.887, 0.0, -0.462));
  let T = normalize(windWorld - N * dot(windWorld, N));
  let B = normalize(cross(N, T));
  let alongSlope = dot(H, T) / ndh;
  let acrossSlope = dot(H, B) / ndh;
  // Cox-Munk clean-sea mean-square slopes at an 11.5 m/s wind.
  let alongVariance = 0.0363;
  let acrossVariance = 0.0251;
  let slopePdf = exp(-0.5 * (alongSlope * alongSlope / alongVariance + acrossSlope * acrossSlope / acrossVariance))
    / (6.2831853 * sqrt(alongVariance * acrossVariance));
  let facetDistribution = slopePdf / max(ndh * ndh * ndh * ndh, 0.0001);
  let visibility = smithVisibility(ndv, 0.0307) * smithVisibility(ndl, 0.0307);
  return dielectricFresnel(max(dot(V, H), 0.0)) * facetDistribution * visibility / max(4.0 * ndv, 0.001);
}

fn breakerFrontPosition(time: f32) -> f32 {
  let travellingPhase = time * 2.4 + 12.0;
  return travellingPhase - floor(travellingPhase / 72.0) * 72.0 - 36.0;
}

fn breakerFrontVisibility(front: f32) -> f32 {
  // Fade the front out before its periodic reset, then reintroduce it from the
  // opposite side. This avoids a 72 m position pop in both geometry and the
  // adaptive sampling warp.
  return 1.0 - smoothstep(28.0, 35.0, abs(front));
}

fn breakerEventActivation(along: f32) -> f32 {
  let uv = vec2<f32>(clamp(along / 360.0 + 0.5, 0.0, 1.0), 0.5);
  return smoothstep(0.035, 0.68, textureSampleLevel(breaker_events, field_sampler, uv, 0.0).r);
}

fn breakerCoordinates(p: vec2<f32>, time: f32) -> vec2<f32> {
  let travelDirection = normalize(vec2<f32>(0.887, -0.462));
  let tangentDirection = vec2<f32>(-travelDirection.y, travelDirection.x);
  let along = dot(p, tangentDirection);
  let meander = sin(along * 0.055 + time * 0.055 + 0.7) * 3.8
    + sin(along * 0.14 - time * 0.032 - 1.3) * 1.2;
  let signedDistance = dot(p, travelDirection) - breakerFrontPosition(time) - meander;
  return vec2<f32>(signedDistance, along);
}

fn adaptiveBreakerCoordinates(p: vec2<f32>, time: f32) -> vec2<f32> {
  // Concentrate the existing uniform-grid samples around the moving front.
  // The linear compensation makes the warp approach zero at the domain edge,
  // so this is a redistribution of vertices rather than an expanding patch.
  let travelDirection = normalize(vec2<f32>(0.887, -0.462));
  let tangentDirection = vec2<f32>(-travelDirection.y, travelDirection.x);
  let across = dot(p, travelDirection);
  let along = dot(p, tangentDirection);
  let front = breakerFrontPosition(time);
  let domainHalfDiagonal = 276.0;
  let concentration = 8.2 * breakerFrontVisibility(front) * breakerEventActivation(along);
  let bandWidth = 12.5;
  let correction = -concentration * tanh((across - front) / bandWidth)
    + concentration * across / domainHalfDiagonal;
  return p + travelDirection * correction;
}

fn localizedBreakerDisplacement(p: vec2<f32>, time: f32) -> vec4<f32> {
  // A travelling, meandering nonlinear wavefront blended into the same water
  // parameterization. Unlike the rejected detached crest sheet, its edges
  // converge to the spectral surface and its horizontal motion can fold the
  // grid naturally when the crest becomes steep.
  let travelDirection = normalize(vec2<f32>(0.887, -0.462));
  let breakerCoord = breakerCoordinates(p, time);
  let along = breakerCoord.y;
  let travellingFront = breakerFrontPosition(time);
  let frontVisibility = breakerFrontVisibility(travellingFront);
  let signedDistance = breakerCoord.x;
  let u = signedDistance / 9.0;
  let edgeWindow = 1.0 - smoothstep(0.82, 1.34, abs(u));
  let localEnvelope = exp(-0.55 * u * u) * edgeWindow;
  let phase = 3.14159265 * u;
  let crestProfile = cos(phase);
  let alongSignal = sin(along * 0.041 + time * 0.08 + 0.4)
    + 0.48 * sin(along * 0.097 - time * 0.035 - 1.2);
  let alongBreakup = smoothstep(-0.35, 0.72, alongSignal);
  let alongVariation = (0.42 + 0.58 * alongBreakup)
    * (0.90 + 0.10 * sin(along * 0.092 + 1.8));
  let vertical = 2.45 * localEnvelope * crestProfile * alongVariation;
  var horizontal = -travelDirection * 2.15 * localEnvelope * sin(phase) * alongVariation;
  let lip = pow(max(crestProfile, 0.0), 3.0) * localEnvelope;
  horizontal += travelDirection * 0.72 * lip * alongVariation;
  let activation = breakerEventActivation(along);
  return vec4<f32>(horizontal.x, vertical, horizontal.y, lip) * frontVisibility * activation * (1.0 - uniforms.environment.x);
}

fn evaluateWaterSurface(p: vec2<f32>) -> SurfaceEvaluation {
  let fieldUv = clamp(p / uniforms.terrain.x + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
  let terrain = textureSampleLevel(terrain_field, field_sampler, fieldUv, 0.0);
  let depth = uniforms.sunWater.w - terrain.r;
  let shallowAttenuation = smoothstep(0.14, 2.7, depth);
  let simUv = (p - uniforms.simulation.xy) / uniforms.simulation.z + vec2<f32>(0.5);
  let longUv = fract(p / 240.0 + vec2<f32>(0.5));
  let mediumUv = fract(p / 64.0 + vec2<f32>(0.5));
  let long0 = textureSampleLevel(long_field0, spectrum_sampler, longUv, 0.0);
  let long1 = textureSampleLevel(long_field1, spectrum_sampler, longUv, 0.0);
  let medium0 = textureSampleLevel(medium_field0, spectrum_sampler, mediumUv, 0.0);
  let medium1 = textureSampleLevel(medium_field1, spectrum_sampler, mediumUv, 0.0);
  let horizontalDisplacement = long0.rg * 1.18 + medium0.rg * 1.05;
  let longHeight = long0.b;
  let mediumHeight = medium0.b;
  let spectralHeight = longHeight + mediumHeight
    + 0.14 * (longHeight * longHeight - 0.080)
    + 0.32 * (mediumHeight * mediumHeight - 0.030);
  let crossDerivative = long0.a * 1.18 + medium0.a * 1.05;
  let longSlope = long1.rg * (1.0 + 0.28 * longHeight);
  let mediumSlope = medium1.rg * (1.0 + 0.64 * mediumHeight);
  let spectralSlope = longSlope + mediumSlope;
  let horizontalDerivative = long1.ba * 1.18 + medium1.ba * 1.05;
  let sim = simulationSample(simUv);
  let texel = uniforms.simulation.w;
  let left = simulationSample(simUv - vec2<f32>(texel, 0.0)).r;
  let right = simulationSample(simUv + vec2<f32>(texel, 0.0)).r;
  let back = simulationSample(simUv - vec2<f32>(0.0, texel)).r;
  let front = simulationSample(simUv + vec2<f32>(0.0, texel)).r;
  let worldTexel = uniforms.simulation.z * texel;
  let simulationDerivative = vec2<f32>(right - left, front - back) / max(worldTexel * 2.0, 0.001);
  let simulationEdge = min(min(simUv.x, 1.0 - simUv.x), min(simUv.y, 1.0 - simUv.y));
  let simulationCoverage = step(0.0, simulationEdge) * smoothstep(0.008, 0.055, simulationEdge);
  let baseJacobian = (1.0 + horizontalDerivative.x) * (1.0 + horizontalDerivative.y) - crossDerivative * crossDerivative;
  let breaker = localizedBreakerDisplacement(p, uniforms.cameraTime.w) * shallowAttenuation;
  let breakerStep = 0.55;
  let breakerDx = (localizedBreakerDisplacement(p + vec2<f32>(breakerStep, 0.0), uniforms.cameraTime.w) * shallowAttenuation - breaker) / breakerStep;
  let breakerDz = (localizedBreakerDisplacement(p + vec2<f32>(0.0, breakerStep), uniforms.cameraTime.w) * shallowAttenuation - breaker) / breakerStep;
  // The nonlinear field is a replacement for the far FFT within its domain,
  // not an additive wake texture. Its relaxation band already matches the FFT,
  // and this narrow geometric blend hides the finite-domain edge.
  let nearshoreOwnership = simulationCoverage * (1.0 - smoothstep(3.8, 5.55, depth));
  let wave = mix(spectralHeight * shallowAttenuation, sim.r, nearshoreOwnership) + breaker.y;
  let world = vec3<f32>(
    p.x + horizontalDisplacement.x * shallowAttenuation + breaker.x,
    uniforms.sunWater.w + wave,
    p.y + horizontalDisplacement.y * shallowAttenuation + breaker.z
  );
  let blendedSlope = mix(spectralSlope * shallowAttenuation, simulationDerivative, nearshoreOwnership);
  let tangentPX = vec3<f32>(1.0 + horizontalDerivative.x * shallowAttenuation + breakerDx.x, blendedSlope.x + breakerDx.y, crossDerivative * shallowAttenuation + breakerDx.z);
  let tangentPZ = vec3<f32>(crossDerivative * shallowAttenuation + breakerDz.x, blendedSlope.y + breakerDz.y, 1.0 + horizontalDerivative.y * shallowAttenuation + breakerDz.z);
  var result: SurfaceEvaluation;
  result.world = world;
  result.tangentPX = tangentPX;
  result.tangentPZ = tangentPZ;
  result.fieldUv = fieldUv;
  result.simulationUv = simUv;
  result.waveHeight = wave;
  result.compression = max(0.0, 1.0 - baseJacobian) * shallowAttenuation + breaker.w * 0.38;
  result.breakerLip = breaker.w;
  return result;
}

@vertex fn waterVertex(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) instanceId: u32) -> WaterOutput {
  var corners = array<vec2<u32>, 6>(
    vec2<u32>(0u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u, 0u),
    vec2<u32>(0u, 1u), vec2<u32>(1u, 1u), vec2<u32>(1u, 0u)
  );
  let shoreScene = uniforms.environment.x > 0.5;
  let resolution = select(64u, u32(uniforms.environment.z), shoreScene);
  let cellId = vertexId / 6u;
  let cell = vec2<u32>(cellId % resolution, cellId / resolution);
  let grid = cell + corners[vertexId % 6u];
  let uv = vec2<f32>(grid) / f32(resolution);
  var baseP = (uv - vec2<f32>(0.5)) * uniforms.terrain.x;
  if (!shoreScene) {
    let level = f32(instanceId);
    let halfExtent = 32.0 * exp2(level);
    let cellSize = halfExtent * 2.0 / f32(resolution);
    let snappedCamera = floor(uniforms.cameraTime.xz / cellSize) * cellSize;
    baseP = snappedCamera + (uv - vec2<f32>(0.5)) * halfExtent * 2.0;
    if (instanceId > 0u) {
      // Degenerate the covered centre of each coarser level. A one-cell
      // underlap keeps T-junctions hidden while all ring origins stay snapped.
      let innerHalf = halfExtent * 0.5 - cellSize;
      let cellCenter = snappedCamera + ((vec2<f32>(cell) + vec2<f32>(0.5)) / f32(resolution) - vec2<f32>(0.5)) * halfExtent * 2.0;
      if (all(abs(cellCenter - snappedCamera) < vec2<f32>(innerHalf))) {
        baseP = vec2<f32>(10000.0);
      }
    }
  }
  let coordinateStep = 0.55;
  let p = adaptiveBreakerCoordinates(baseP, uniforms.cameraTime.w);
  let pDx = (adaptiveBreakerCoordinates(baseP + vec2<f32>(coordinateStep, 0.0), uniforms.cameraTime.w) - p) / coordinateStep;
  let pDz = (adaptiveBreakerCoordinates(baseP + vec2<f32>(0.0, coordinateStep), uniforms.cameraTime.w) - p) / coordinateStep;
  let surface = evaluateWaterSurface(p);
  let tangentX = surface.tangentPX * pDx.x + surface.tangentPZ * pDx.y;
  let tangentZ = surface.tangentPX * pDz.x + surface.tangentPZ * pDz.y;
  let breakerCoord = breakerCoordinates(p, uniforms.cameraTime.w);
  var output: WaterOutput;
  output.position = uniforms.viewProj * vec4<f32>(surface.world, 1.0);
  output.world = surface.world;
  output.normal = normalize(cross(tangentZ, tangentX));
  output.fieldUv = surface.fieldUv;
  output.simulationUv = surface.simulationUv;
  output.waveHeight = surface.waveHeight;
  output.compression = surface.compression;
  output.breakerLip = surface.breakerLip;
  output.breakerCoord = breakerCoord;
  output.surfaceKind = 0.0;
  return output;
}

fn breakerPatchBreakup(along: f32, time: f32) -> f32 {
  let breakupSignal = sin(along * 0.041 + time * 0.08 + 0.4)
    + 0.48 * sin(along * 0.097 - time * 0.035 - 1.2);
  return smoothstep(-0.10, 0.65, breakupSignal);
}

fn breakerPatchExtra(across: f32, along: f32, time: f32) -> vec3<f32> {
  let travelDirection = normalize(vec2<f32>(0.887, -0.462));
  let frontVisibility = breakerFrontVisibility(breakerFrontPosition(time));
  let u = across / 9.0;
  let edgeWindow = 1.0 - smoothstep(0.70, 1.24, abs(u));
  let alongWindow = 1.0 - smoothstep(158.0, 176.0, abs(along));
  let envelope = exp(-0.58 * u * u) * edgeWindow * alongWindow * frontVisibility * breakerEventActivation(along) * (1.0 - uniforms.environment.x);
  let phase = 3.14159265 * u;
  let alongVariation = 0.86 + 0.14 * sin(along * 0.092 + 1.8);
  let breakup = 0.24 + 0.76 * breakerPatchBreakup(along, time);
  let lip = pow(max(cos(phase), 0.0), 3.0);
  // A narrow crest-nose correction rather than another full wave profile.
  // The base spectral/bound-harmonic surface owns the body of the wave; this
  // only rounds and leans locally breaking sections without forming a shelf.
  let horizontalAmount = 0.62 * breakup * lip * envelope * alongVariation;
  let vertical = 0.14 * breakup * lip * envelope * alongVariation;
  return vec3<f32>(travelDirection.x * horizontalAmount, vertical, travelDirection.y * horizontalAmount);
}

@vertex fn breakerPatchVertex(@builtin(vertex_index) vertexId: u32) -> WaterOutput {
  var corners = array<vec2<u32>, 6>(
    vec2<u32>(0u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u, 0u),
    vec2<u32>(0u, 1u), vec2<u32>(1u, 1u), vec2<u32>(1u, 0u)
  );
  let alongResolution = 256u;
  let acrossResolution = 48u;
  let cellId = vertexId / 6u;
  let cell = vec2<u32>(cellId % alongResolution, cellId / alongResolution);
  let grid = cell + corners[vertexId % 6u];
  let uv = vec2<f32>(grid) / vec2<f32>(f32(alongResolution), f32(acrossResolution));
  let along = mix(-180.0, 180.0, uv.x);
  let across = mix(-12.0, 12.0, uv.y);
  let travelDirection = normalize(vec2<f32>(0.887, -0.462));
  let tangentDirection = vec2<f32>(-travelDirection.y, travelDirection.x);
  let time = uniforms.cameraTime.w;
  let meander = sin(along * 0.055 + time * 0.055 + 0.7) * 3.8
    + sin(along * 0.14 - time * 0.032 - 1.3) * 1.2;
  let meanderDerivative = cos(along * 0.055 + time * 0.055 + 0.7) * 0.209
    + cos(along * 0.14 - time * 0.032 - 1.3) * 0.168;
  let p = tangentDirection * along + travelDirection * (breakerFrontPosition(time) + meander + across);
  let pAlong = tangentDirection + travelDirection * meanderDerivative;
  let pAcross = travelDirection;
  let surface = evaluateWaterSurface(p);
  let extra = breakerPatchExtra(across, along, time);
  let derivativeStep = 0.12;
  let extraAlong = (breakerPatchExtra(across, along + derivativeStep, time) - breakerPatchExtra(across, along - derivativeStep, time)) / (2.0 * derivativeStep);
  let extraAcross = (breakerPatchExtra(across + derivativeStep, along, time) - breakerPatchExtra(across - derivativeStep, along, time)) / (2.0 * derivativeStep);
  let tangentAlong = surface.tangentPX * pAlong.x + surface.tangentPZ * pAlong.y + extraAlong;
  let tangentAcross = surface.tangentPX * pAcross.x + surface.tangentPZ * pAcross.y + extraAcross;
  let world = surface.world + extra;
  var output: WaterOutput;
  output.position = uniforms.viewProj * vec4<f32>(world, 1.0);
  output.world = world;
  output.normal = normalize(cross(tangentAlong, tangentAcross));
  output.fieldUv = surface.fieldUv;
  output.simulationUv = surface.simulationUv;
  output.waveHeight = surface.waveHeight + extra.y;
  let patchBreakup = breakerPatchBreakup(along, time);
  output.compression = surface.compression + smoothstep(0.62, 0.94, patchBreakup) * (1.0 - smoothstep(0.3, 4.5, abs(across))) * 0.055;
  output.breakerLip = surface.breakerLip;
  output.breakerCoord = vec2<f32>(across, along);
  output.surfaceKind = 1.0;
  return output;
}

@fragment fn waterFragment(input: WaterOutput) -> @location(0) vec4<f32> {
  let patchVisible = breakerFrontVisibility(breakerFrontPosition(uniforms.cameraTime.w)) * (1.0 - uniforms.environment.x);
  let patchAlong = 1.0 - smoothstep(158.0, 176.0, abs(input.breakerCoord.y));
  if (input.surfaceKind < 0.5 && patchVisible * patchAlong > 0.001 && abs(input.breakerCoord.x) < 11.72) { discard; }
  if (input.surfaceKind > 0.5 && (patchVisible <= 0.001 || abs(input.breakerCoord.x) > 11.82 || patchAlong <= 0.001)) { discard; }
  let state = simulationSample(input.simulationUv);
  let displacedTerrainUv = input.world.xz / uniforms.terrain.x + vec2<f32>(0.5);
  if (any(displacedTerrainUv < vec2<f32>(0.0)) || any(displacedTerrainUv > vec2<f32>(1.0))) { discard; }
  let terrain = textureSample(terrain_field, field_sampler, displacedTerrainUv);
  // Coverage must be derived from the same displaced surface that produced
  // the raster depth. Re-evaluating height per fragment makes colour and depth
  // disagree at wet/dry intersections, exposing a checkerboard of triangles.
  let waterColumn = input.world.y - terrain.r;
  let shorelineWidth = clamp(fwidth(waterColumn), 0.006, 0.06);
  // Leave a centimetre-scale wet-sand margin in the island scene. Rendering
  // translucent water almost coplanar with terrain is visually unstable and
  // was the remaining source of dotted/checker shoreline fragments.
  let shorelineThreshold = mix(0.018, 0.28, uniforms.environment.x);
  let shorelineCoverage = smoothstep(shorelineThreshold - shorelineWidth, shorelineThreshold + shorelineWidth, waterColumn);
  if (shorelineCoverage < 0.01) { discard; }
  let depth = max(waterColumn, 0.018);
  let p = input.world.xz;
  let time = uniforms.cameraTime.w;
  let shortUv = fract(p / 12.0 + vec2<f32>(0.5));
  let short0 = textureSample(short_field0, spectrum_sampler, shortUv);
  let short1 = textureSample(short_field1, spectrum_sampler, shortUv);
  let shortDistanceFade = 1.0 - smoothstep(42.0, 118.0, distance(uniforms.cameraTime.xyz, input.world));
  let shortSlope = short1.rg * shortDistanceFade;
  // Short waves become an aggregate slope distribution instead of a literal
  // high-frequency normal texture. This is the geometry-to-BRDF transition
  // used to avoid sparkling/streaking as sub-pixel waves recede.
  var N = normalize(input.normal + vec3<f32>(-shortSlope.x, 0.0, -shortSlope.y) * 0.42);
  let surfaceRoughness = mix(0.035, 0.115, smoothstep(0.012, 0.30, length(shortSlope)));
  let underwater = uniforms.terrain.w > 0.5;
  if (underwater) { N *= -1.0; }
  let V = normalize(uniforms.cameraTime.xyz - input.world);
  let L = normalize(uniforms.sunWater.xyz);
  let ndv = clamp(abs(dot(N, V)), 0.0, 1.0);
  let fresnel = dielectricFresnel(ndv);
  let terrainNormalY = sqrt(max(1.0 - terrain.g * terrain.g - terrain.b * terrain.b, 0.0001));
  let floorLight = clamp(dot(normalize(vec3<f32>(terrain.g, terrainNormalY, terrain.b)), L) * 0.56 + 0.48, 0.0, 1.0);
  let refractedOffset = N.xz * depth * mix(0.42, 1.25, 1.0 - ndv);
  let refractedP = p + refractedOffset;
  let sandVariation = valueNoise(refractedP * 0.18) * 0.055 + valueNoise(refractedP * 0.62 + vec2<f32>(7.1, -3.4)) * 0.018;
  var floorColor = (vec3<f32>(0.46, 0.37, 0.225) + vec3<f32>(sandVariation)) * mix(0.70, 1.02, floorLight);
  // Open-ocean build: the shore scene captured color/depth path is removed,
  // the procedural seabed color is always used directly.
  var capturedLinear = floorColor;
  let opticalDepth = select(depth / max(ndv, 0.28), max(0.0, uniforms.sunWater.w - uniforms.cameraTime.y) / max(abs(dot(N, V)), 0.32), underwater);
  let absorption = vec3<f32>(0.37, 0.125, 0.054);
  let transmission = exp(-absorption * min(opticalDepth, 24.0));
  // Open water is not a cyan diffuse material.  Keep the in-scattered body
  // colour low-energy so the interface reflection supplies the bright values.
  let scatterColor = vec3<f32>(0.0035, 0.096, 0.092);
  let scatterAmount = vec3<f32>(1.0) - transmission;
  let phaseG = 0.24;
  let lightCosine = dot(-V, L);
  let phase = (1.0 - phaseG * phaseG) / pow(max(1.0 + phaseG * phaseG - 2.0 * phaseG * lightCosine, 0.04), 1.5);
  var refracted = floorColor * transmission + scatterColor * scatterAmount * (0.72 + phase * 0.060);
  let refractionSoftness = smoothstep(3.0, 15.0, opticalDepth) * 0.08;
  refracted = mix(refracted, scatterColor, refractionSoftness);
  let reflectedDirection = reflect(-V, N);
  var reflected = skyColor(reflectedDirection, time, L);
  // Preserve environment contrast.  Tinting the reflection toward the water
  // body colour was the main source of the previous milky/plastic response.
  reflected *= mix(vec3<f32>(0.70, 0.77, 0.80), vec3<f32>(0.76, 0.81, 0.83), surfaceRoughness);
  let playerDistance = length(p - uniforms.player.xy);
  let nearSwimmer = 1.0 - smoothstep(0.85, 3.4, playerDistance);
  var reflectionWeight = select(fresnel, fresnel * 0.07, underwater);
  let shoreShallows = uniforms.environment.x * (1.0 - smoothstep(0.12, 1.02, depth));
  // At the waterline the captured sand is still the dominant optical path.
  // Suppress the old cyan body-colour halo and retain a thin, green-blue
  // transmission tint instead of treating centimetres of water like ocean.
  let shallowTransmission = capturedLinear * vec3<f32>(0.66, 0.62, 0.52)
    + vec3<f32>(0.004, 0.013, 0.011) * smoothstep(0.10, 0.90, depth);
  refracted = mix(refracted, shallowTransmission, shoreShallows * 0.76);
  reflectionWeight *= 1.0 - shoreShallows * 0.56;
  reflectionWeight *= mix(1.0, 0.38, nearSwimmer * uniforms.interaction.y);
  var color = mix(refracted, reflected, clamp(reflectionWeight, 0.0, 0.92));
  // A height-based colour wash made crests look like translucent resin.  The
  // spectral normal, Fresnel response and actual foam now carry that contrast.
  let velocityDirection = normalize(uniforms.player.zw + vec2<f32>(0.0001, 0.0));
  let toPlayer = p - uniforms.player.xy;
  let behind = smoothstep(-0.2, 2.8, dot(toPlayer, -velocityDirection));
  let wakeRibbon = exp(-pow(abs(dot(toPlayer, vec2<f32>(-velocityDirection.y, velocityDirection.x))) / 0.64, 2.0)) * (1.0 - smoothstep(0.8, 6.5, playerDistance)) * behind;
  let wake = wakeRibbon * smoothstep(0.5, 5.5, uniforms.interaction.x) * uniforms.interaction.y;
  let crestHeight = smoothstep(0.27, 0.72, input.waveHeight);
  let shortCrossDerivative = short0.a * 0.40;
  let shortHorizontalDerivative = short1.ba * 0.40;
  let shortJacobian = (1.0 + shortHorizontalDerivative.x) * (1.0 + shortHorizontalDerivative.y) - shortCrossDerivative * shortCrossDerivative;
  let surfaceCompression = input.compression + max(0.0, 1.0 - shortJacobian) * shortDistanceFade * 0.62;
  let crestPinch = smoothstep(0.16, 0.34, surfaceCompression);
  let crestVariation = valueNoise(p * 0.37 + vec2<f32>(time * 0.021, -time * 0.016)) * 0.55
    + valueNoise(p * 1.41 + vec2<f32>(-time * 0.043, time * 0.032)) * 0.30
    + valueNoise(p * 3.7 + vec2<f32>(time * 0.081, -time * 0.066)) * 0.15;
  let crestBreakup = smoothstep(0.60, 0.80, crestVariation);
  let crestDistanceFade = 1.0 - smoothstep(95.0, 188.0, distance(uniforms.cameraTime.xyz, input.world));
  let breakerBreakup = smoothstep(0.43, 0.62, crestVariation);
  let breakerFoam = smoothstep(0.24, 0.72, input.breakerLip) * breakerBreakup * crestDistanceFade;
  let whitecap = max(crestHeight * pow(crestPinch, 4.0) * crestBreakup, breakerFoam * 0.78) * crestDistanceFade;
  let persistentBreakup = 0.48
    + valueNoise(p * 0.72 + vec2<f32>(time * 0.012, -time * 0.009)) * 0.34
    + valueNoise(p * 2.45 + vec2<f32>(-time * 0.024, time * 0.018)) * 0.18;
  let foamBreakup = smoothstep(0.46, 0.78, persistentBreakup);
  let persistentFoam = state.a * 0.58 * foamBreakup * (1.0 - uniforms.environment.x);
  // The coastal wash is selected by the conservative nearshore state rather
  // than by a decorative noise strip. Momentum makes active run-up brighter;
  // the depth window lets it naturally retreat with the simulated waterline.
  let nearshoreSpeed = length(state.gb) / max(depth, 0.08);
  let swashDepth = smoothstep(0.035, 0.11, depth) * (1.0 - smoothstep(0.24, 0.62, depth));
  let activeSwash = smoothstep(0.018, 0.24, state.a) * clamp(0.46 + nearshoreSpeed * 0.12, 0.46, 0.88);
  let shoreStateFoam = uniforms.environment.x * activeSwash * swashDepth * foamBreakup * 0.62;
  var foam = max(max(max(persistentFoam, shoreStateFoam), wake * 0.16), whitecap);
  let visibleFoam = mix(smoothstep(0.16, 0.66, foam), smoothstep(0.055, 0.40, foam), uniforms.environment.x);
  foam = select(visibleFoam, 0.0, underwater);
  let foamCoverage = max(clamp(foam * 0.21, 0.0, 0.145), breakerFoam * 0.22);
  color = mix(color, vec3<f32>(0.80, 0.88, 0.84), clamp(foamCoverage, 0.0, 0.22));
  let sunGlitter = oceanSunGlitter(N, V, L);
  color += vec3<f32>(1.0, 0.91, 0.70) * min(sunGlitter * 0.070, 0.34) * select(1.0, 0.08, underwater);
  if (underwater) {
    let viewDepth = min(distance(uniforms.cameraTime.xyz, input.world), 22.0);
    color = mix(color, vec3<f32>(0.012, 0.205, 0.190), 0.42 + viewDepth / 22.0 * 0.12);
  }
  return vec4<f32>(linearToSrgb(aces(color)), shorelineCoverage);
}
