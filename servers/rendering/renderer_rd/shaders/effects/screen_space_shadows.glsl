#[compute]

#version 450

#VERSION_DEFINES

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba8, set = 0, binding = 0) uniform image2D color_image;
layout(set = 1, binding = 0) uniform sampler2D depth_image;

layout(set = 2, binding = 0, std140) uniform Params {
    mat4x4 proj;
    mat4x4 proj_inv;
    mat4x4 view;
    mat4x4 view_inv;
} params;

layout(push_constant, std430) uniform PushConstant {
    vec2 screen_size_rcp;
    ivec2 screen_size;

    vec3 light_dir;
    float thickness;

    float max_dist;
    float opacity;
    int pad[2];
} push_constant;

float get_noise_interleaved(vec2 screen_pos) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(screen_pos, magic.xy)));
}

float get_noise_interleaved2(vec2 pixCoord)
{
    const vec3 magic = vec3(0.06711056f, 0.00583715f, 52.9829189f);
    vec2 frameMagicScale = vec2(2.083f, 4.867f);
    return fract(magic.z * fract(dot(pixCoord, magic.xy)));
}

vec4 world_to_clip(vec3 world_pos, mat4 view, mat4 proj) {
    vec4 view_pos = view * vec4(world_pos, 1.0);
    return proj * view_pos;
}

vec4 world_to_clip_dir(vec3 world_pos, mat4 view, mat4 proj) {
    vec4 view_pos = view * vec4(world_pos, 0.0);
    return proj * view_pos;
}

vec4 screen_to_world(vec2 uv, float depth, mat4 invproj, mat4 invview) {
    vec3 ndc = vec3(uv * 2.0 - 1.0, depth);
    vec4 vs = invproj * vec4(ndc, 1.0);
    return invview * vec4((vs.xyz / vs.w), 1.0);
}

vec4 screen_to_clip(vec2 uv, float depth, mat4 invproj) {
    vec3 ndc = vec3(uv * 2.0 - 1.0, depth);
    vec4 vs = invproj * vec4(ndc, 1.0);
    return vec4((vs.xyz / vs.w), 1.0);
}

bool is_valid_uv(vec2 uv) {
    return all(greaterThanEqual(uv, vec2(0.0))) && all(lessThan(uv, vec2(1.0)));
}

float screen_space_shadow_raycast(vec3 position_ws, vec3 ray_dir_ws, float initial_depth, float linear_depth, float ray_length, vec2 position_ss, mat4 viewmat, mat4 projmat, mat4 invviewmat, mat4 invprojmat, out float fade) {

    float sss_smoothness = 0.0;
    float ray_bias = 0.1 * 0.0001;

    ray_length = ray_length * max(0.5, linear_depth * /*_ContactShadowDistanceScaleFactor*/0.5);

    if (initial_depth < 0.0001)
    {
        return 1.0;
    }
    float dither_bias = 0.5;
    int sample_count = 16;
    float dither = get_noise_interleaved2(position_ss * push_constant.screen_size) - dither_bias;

    vec3 ray_start_ws = position_ws - position_ws * ray_bias;
    vec3 ray_end_ws = ray_start_ws + ray_dir_ws * ray_length;

    vec4 ray_start_cs = world_to_clip(ray_start_ws, viewmat, projmat);
    vec4 ray_end_cs = world_to_clip(ray_end_ws, viewmat, projmat);

    // Calculate orthogonal ray for threshold computation
    vec4 ray_ortho_cs = ray_start_cs + vec4(projmat[0][2],
                                            projmat[1][2],
                                            projmat[2][2],
                                            projmat[3][2]) * ray_length;
    ray_ortho_cs.xyz /= ray_ortho_cs.w;
    ray_start_cs.xyz /= ray_start_cs.w;
    ray_end_cs.xyz /= ray_end_cs.w;

    vec3 ray_dir_cs = ray_end_cs.xyz - ray_start_cs.xyz;

    float step_size = 1.0 / float(sample_count);
    float compare_threshold = abs(ray_ortho_cs.z - ray_start_cs.z) * /*_ContactShadowThickness*/push_constant.thickness * 10. * max(0.07, step_size);

    float occluded = 0.;

    vec2 start_uv = ray_start_cs.xy * 0.5 + 0.5;
    vec3 ray_start = vec3(start_uv, ray_start_cs.z);
    vec3 ray_dir = vec3(ray_dir_cs.x * 0.5, ray_dir_cs.y * 0.5, ray_dir_cs.z);

    float t = step_size * dither + step_size;
    // float t = step_size + step_size;

    for (int i = 0; i < sample_count; i++) {
        vec3 sample_pos = ray_start + t * ray_dir;

        if (!is_valid_uv(sample_pos.xy)) {
            break;
        }

        float sample_depth = texture(depth_image, sample_pos.xy).x;

        float depth_diff = sample_depth - sample_pos.z;

        if (depth_diff > 0.0 && abs(compare_threshold - depth_diff) < compare_threshold && sample_pos.z > 0.0) {
            // return smoothstep(sss_smoothness, 1.0, (float(i) / float(sample_count))); // Occluded
            occluded = 1.0f;
            break;
        }

        t += step_size;
    }

    vec2 vignette = max(6.0f * abs(ray_start_cs.xy + ray_dir_cs.xy * t) - 5.0f, 0.0f);
    fade = occluded;
    fade *= clamp(1.0f - dot(vignette, vignette), 0., 1.);

    return occluded; // Not occluded
}

void main() {
    ivec2 iuv = ivec2(gl_GlobalInvocationID.xy);

   if (any(greaterThanEqual(iuv, push_constant.screen_size))) {
       return;
   }

    vec2 uv = (vec2(iuv) + 0.5f) * push_constant.screen_size_rcp;
    vec4 orig_color = imageLoad(color_image, iuv);
    vec3 shadow_color = vec3(0, 0, 0);

    float depth = texture(depth_image, uv).r;
    vec4 roview = vec4(uv * 2. - 1., depth, 1.0);
    roview = params.proj_inv * roview;
    roview.xyz = roview.xyz / roview.w;
    vec3 ro = (params.view_inv * vec4(roview.xyz, 1.0)).xyz;
    vec3 rd = -push_constant.light_dir;
    float fade = 0.;
    float shadow = screen_space_shadow_raycast(ro, rd, depth, -roview.z, push_constant.max_dist, uv, params.view, params.proj, params.view_inv, params.proj_inv, fade);
    //fade *= clamp((-roview.z - /*_ContactShadowMinDistance*/0.) * (/*_ContactShadowFadeInEnd*/1./0.), 0., 1.);
    fade *= clamp((/*_ContactShadowFadeEnd*/50. - (-roview.z)) * /*_ContactShadowFadeOneOverRange*/0.2, 0., 1.);

    imageStore(color_image, iuv, vec4(mix(shadow_color, orig_color.rgb, clamp(1.-fade*shadow, 1.-push_constant.opacity, 1.0)), orig_color.a));
    // imageStore(color_image, iuv, vec4(vec3(clamp(params.proj[2][3]*1000., 0., 1.0)), orig_color.a));
    // imageStore(color_image, iuv, vec4(vec3(clamp(shadow, 0., 1.)), 1.0));
    //imageStore(color_image, iuv, vec4(orig_color.rgb * max(0.2, shadow), orig_color.a));
}
