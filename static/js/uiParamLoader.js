/**
 * uiParamLoader.js
 * 负责渲染各个图像处理动作对应的前端参数输入组件
 */

function renderParams(action) {
    const paramContainer = document.getElementById('paramContainer');
    let html = '';
    
    switch(action) {
        case 'mean_filter':
        case 'median_filter':
        case 'gaussian_blurring':
            html = `<div class="control-group">
                    <label>模糊强度/核大小 <span class="param-value" id="kernelVal">3</span></label>
                    <input type="range" id="param_kernel_size" min="3" max="21" step="2" value="3" oninput="document.getElementById('kernelVal').innerText=this.value">
                </div>`;
            break;
        case 'rotate':
            html = `<div class="control-group">
                    <label>角度 <span class="param-value" id="angleVal">90°</span></label>
                    <input type="range" id="param_angle" min="-180" max="180" step="1" value="90" oninput="document.getElementById('angleVal').innerText=this.value+'°'">
                </div>`;
            break;
        case 'translate':
            html = `<div class="control-group">
                    <label>X偏移 <span class="param-value" id="txVal">0</span></label>
                    <input type="range" id="param_tx" min="-200" max="200" step="1" value="0" oninput="document.getElementById('txVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>Y偏移 <span class="param-value" id="tyVal">0</span></label>
                    <input type="range" id="param_ty" min="-200" max="200" step="1" value="0" oninput="document.getElementById('tyVal').innerText=this.value">
                </div>`;
            break;
        case 'zoom':
            html = `<div class="control-group">
                    <label>X轴缩放 <span class="param-value" id="zxVal">1.0</span></label>
                    <input type="range" id="param_z_x" min="0.1" max="3.0" step="0.1" value="1.0" oninput="document.getElementById('zxVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>Y轴缩放 <span class="param-value" id="zyVal">1.0</span></label>
                    <input type="range" id="param_z_y" min="0.1" max="3.0" step="0.1" value="1.0" oninput="document.getElementById('zyVal').innerText=this.value">
                </div>`;
            break;
        case 'frosted':
            html = `<div class="control-group">
                    <label>磨砂颗粒度 <span class="param-value" id="offsetVal">2</span></label>
                    <input type="range" id="param_offset" min="1" max="15" step="1" value="2" oninput="document.getElementById('offsetVal').innerText=this.value">
                </div>`;
            break;
        case 'binarize':
            html = `<div class="control-group">
                    <label>二值化阈值 <span class="param-value" id="threshVal">127</span></label>
                    <input type="range" id="param_threshold" min="0" max="255" step="1" value="127" oninput="document.getElementById('threshVal').innerText=this.value">
                </div>`;
            break;
        case 'adjust_brightness':
            html = `<div class="control-group">
                    <label>亮度偏移 <span class="param-value" id="briVal">0</span></label>
                    <input type="range" id="param_beta" min="-100" max="100" step="1" value="0" oninput="document.getElementById('briVal').innerText=this.value">
                </div>`;
            break;
        case 'adjust_contrast':
        case 'adjust_saturation':
            html = `<div class="control-group">
                    <label>调整比例因子 <span class="param-value" id="alphaVal">1.0</span></label>
                    <input type="range" id="param_alpha" min="0.1" max="3.0" step="0.1" value="1.0" oninput="document.getElementById('alphaVal').innerText=this.value">
                </div>`;
            break;
        case 'adjust_sharpness':
            html = `<div class="control-group">
                    <label>锐化强度 <span class="param-value" id="amountVal">1.0</span></label>
                    <input type="range" id="param_amount" min="0.1" max="3.0" step="0.1" value="1.0" oninput="document.getElementById('amountVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>细节半径 <span class="param-value" id="radiusVal">3</span></label>
                    <input type="range" id="param_radius" min="1" max="9" step="2" value="3" oninput="document.getElementById('radiusVal').innerText=this.value">
                </div>`;
            break;
        case 'horizontal_collage':
        case 'vertical_collage':
        case 'stitch_images':
        case 'stitch_images_classic':
            html = `<div class="control-group">
                    <label>第二张图像 (必选)</label>
                    <input type="file" id="param_second_image" accept="image/*" style="display:block; margin-top:8px;">
                </div>`;
            if (action !== 'stitch_images' && action !== 'stitch_images_classic') {
                html += `<div class="control-group">
                    <label>拼图间隙 <span class="param-value" id="gapVal">0 px</span></label>
                    <input type="range" id="param_gap" min="0" max="50" step="1" value="0" oninput="document.getElementById('gapVal').innerText=this.value+' px'">
                </div>`;
            }
            break;
        case 'create_reflection':
            html = `<div class="control-group">
                    <label>倒影长度比例 <span class="param-value" id="rrVal">0.5</span></label>
                    <input type="range" id="param_reflection_ratio" min="0.1" max="1.0" step="0.1" value="0.5" oninput="document.getElementById('rrVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>渐变衰减度 <span class="param-value" id="feVal">1.5</span></label>
                    <input type="range" id="param_fade_exponent" min="0.5" max="3.0" step="0.1" value="1.5" oninput="document.getElementById('feVal').innerText=this.value">
                </div>`;
            break;
        case 'super_resolve':
            html = `<div class="control-group" style="padding-bottom: 10px;">
                    <label><input type="checkbox" id="param_denoise" checked> 预处理双边滤波(防噪点放大)</label>
                </div>
                <div class="control-group">
                    <label><input type="checkbox" id="param_sharpen" checked> 后处理USM锐化(增强边缘)</label>
                </div>`;
            break;
        case 'restore_old_photo':
            html = `<div class="control-group" style="padding-bottom: 8px;">
                    <label><input type="checkbox" id="param_inpaint" checked> 识别并修补划痕(Inpaint)</label>
                </div>
                <div class="control-group" style="padding-bottom: 8px;">
                    <label><input type="checkbox" id="param_denoise" checked> 非局部均值平滑降噪</label>
                </div>
                <div class="control-group" style="padding-bottom: 8px;">
                    <label><input type="checkbox" id="param_contrast" checked> CLAHE 自适应对比度增强</label>
                </div>
                <div class="control-group" style="padding-bottom: 8px;">
                    <label><input type="checkbox" id="param_dehaze"> 褪色/暗通道去雾</label>
                </div>
                <div class="control-group">
                    <label><input type="checkbox" id="param_sharpen" checked> USM 微锐化提取边缘</label>
                </div>`;
            break;
        case 'lowpass_filter':
        case 'highpass_filter':
            html = `<div class="control-group">
                    <label>频率截断半径 Cutoff <span class="param-value" id="cutoffVal">0.2</span></label>
                    <input type="range" id="param_cutoff" min="0.01" max="0.99" step="0.01" value="0.2" oninput="document.getElementById('cutoffVal').innerText=this.value">
                </div>`;
            break;
        case 'bilateral_filter':
            html = `<div class="control-group">
                    <label>滤波核大小/直径 <span class="param-value" id="bkVal">5</span></label>
                    <input type="range" id="param_kernel_size" min="3" max="21" step="2" value="5" oninput="document.getElementById('bkVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>Sigma 色彩空间 <span class="param-value" id="bcVal">15.0</span></label>
                    <input type="range" id="param_sigma_s" min="1" max="150" step="1" value="15" oninput="document.getElementById('bcVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>Sigma 坐标空间 <span class="param-value" id="bsVal">30.0</span></label>
                    <input type="range" id="param_sigma_r" min="1" max="150" step="1" value="30" oninput="document.getElementById('bsVal').innerText=this.value">
                </div>`;
            break;
        case 'bandpass_filter':
        case 'bandreject_filter':
            html = `<div class="control-group">
                    <label>低频截断 Low Cut <span class="param-value" id="lcVal">0.1</span></label>
                    <input type="range" id="param_low_cut" min="0.01" max="0.5" step="0.01" value="0.1" oninput="document.getElementById('lcVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>高频截断 High Cut <span class="param-value" id="hcVal">0.4</span></label>
                    <input type="range" id="param_high_cut" min="0.02" max="0.99" step="0.01" value="0.4" oninput="document.getElementById('hcVal').innerText=this.value">
                </div>`;
            break;
        case 'shear':
            html = `<div class="control-group">
                    <label>左侧裁裁切比例% <span class="param-value" id="slVal">0</span></label>
                    <input type="range" id="param_start_x" min="0" max="49" step="1" value="0" oninput="document.getElementById('slVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>右侧裁剪比例% <span class="param-value" id="srVal">0</span></label>
                    <input type="range" id="param_end_x" min="0" max="49" step="1" value="0" oninput="document.getElementById('srVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>顶部裁剪比例% <span class="param-value" id="stVal">0</span></label>
                    <input type="range" id="param_start_y" min="0" max="49" step="1" value="0" oninput="document.getElementById('stVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>底部裁剪比例% <span class="param-value" id="sbVal">0</span></label>
                    <input type="range" id="param_end_y" min="0" max="49" step="1" value="0" oninput="document.getElementById('sbVal').innerText=this.value">
                </div>`;
            break;
        case 'add_border':
            html = `<div class="control-group">
                    <label>画框模板图片 (必选)</label>
                    <input type="file" id="param_second_image" accept="image/*" style="display:block; margin-top:8px;">
                </div>
                <div class="control-group">
                    <label>照片缩放比例 <span class="param-value" id="bscaleVal">0.1</span></label>
                    <input type="range" id="param_scale" min="0.01" max="1.0" step="0.01" value="0.1" oninput="document.getElementById('bscaleVal').innerText=this.value">
                </div>`;
            break;
        case 'false_color_channel_swap':
            html = `<div class="control-group">
                    <label>R通道 选源 <span class="param-value" id="rsrcVal">2(R)</span></label>
                    <input type="range" id="param_r_src" min="0" max="2" step="1" value="2" oninput="document.getElementById('rsrcVal').innerText=this.value==0?'0(B)':(this.value==1?'1(G)':'2(R)')">
                </div>
                <div class="control-group">
                    <label>G通道 选源 <span class="param-value" id="gsrcVal">0(B)</span></label>
                    <input type="range" id="param_g_src" min="0" max="2" step="1" value="0" oninput="document.getElementById('gsrcVal').innerText=this.value==0?'0(B)':(this.value==1?'1(G)':'2(R)')">
                </div>
                <div class="control-group">
                    <label>B通道 选源 <span class="param-value" id="bsrcVal">1(G)</span></label>
                    <input type="range" id="param_b_src" min="0" max="2" step="1" value="1" oninput="document.getElementById('bsrcVal').innerText=this.value==0?'0(B)':(this.value==1?'1(G)':'2(R)')">
                </div>`;
            break;
        case 'synthesize_false_color_image':
            html = `<div class="control-group">
                    <label>波段 2 图像 (必选)</label>
                    <input type="file" id="param_second_image" accept="image/*" style="display:block; margin-top:8px;">
                </div>
                <div class="control-group">
                    <label>波段 3 图像 (必选)</label>
                    <input type="file" id="param_third_image" accept="image/*" style="display:block; margin-top:8px;">
                </div>
                <div class="control-group">
                    <label>R通道 选源 (0~2) <span class="param-value" id="rsrcVal">2(img3)</span></label>
                    <input type="range" id="param_r_src" min="0" max="2" step="1" value="2" oninput="document.getElementById('rsrcVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>G通道 选源 (0~2) <span class="param-value" id="gsrcVal">0(img1)</span></label>
                    <input type="range" id="param_g_src" min="0" max="2" step="1" value="0" oninput="document.getElementById('gsrcVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>B通道 选源 (0~2) <span class="param-value" id="bsrcVal">1(img2)</span></label>
                    <input type="range" id="param_b_src" min="0" max="2" step="1" value="1" oninput="document.getElementById('bsrcVal').innerText=this.value">
                </div>`;
            break;
        case 'intelligent_fill_light':
            html = `<div class="control-group">
                    <label>补光强度 <span class="param-value" id="iflStrVal">0.6</span></label>
                    <input type="range" id="param_ifl_strength" min="0.1" max="1.0" step="0.1" value="0.6" oninput="document.getElementById('iflStrVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>暗区阈值 <span class="param-value" id="iflThreshVal">0.4</span></label>
                    <input type="range" id="param_ifl_threshold" min="0.1" max="0.9" step="0.1" value="0.4" oninput="document.getElementById('iflThreshVal').innerText=this.value">
                </div>`;
            break;
        case 'adjust_highlight':
            html = `<div class="control-group">
                    <label>增强强度 <span class="param-value" id="ahStrVal">0.5</span></label>
                    <input type="range" id="param_ah_strength" min="0.1" max="1.0" step="0.1" value="0.5" oninput="document.getElementById('ahStrVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>高光阈值 <span class="param-value" id="ahThreshVal">0.7</span></label>
                    <input type="range" id="param_ah_threshold" min="0.5" max="0.95" step="0.05" value="0.7" oninput="document.getElementById('ahThreshVal').innerText=this.value">
                </div>`;
            break;
        case 'add_salt_pepper_noise_optimized':
        case 'add_noise_salt_pepper':
            html = `<div class="control-group">
                    <label>噪声概率 <span class="param-value" id="probVal">0.05</span></label>
                    <input type="range" id="param_prob" min="0.01" max="0.5" step="0.01" value="0.05" oninput="document.getElementById('probVal').innerText=this.value">
                </div>`;
            break;
        case 'add_noise_gaussian':
            html = `<div class="control-group">
                    <label>高斯噪声方差 <span class="param-value" id="varVal">0.01</span></label>
                    <input type="range" id="param_var" min="0.001" max="0.1" step="0.001" value="0.01" oninput="document.getElementById('varVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>高斯噪声均值 <span class="param-value" id="meanVal">0</span></label>
                    <input type="range" id="param_mean" min="-0.5" max="0.5" step="0.01" value="0" oninput="document.getElementById('meanVal').innerText=this.value">
                </div>`;
            break;
        case 'smart_sharpen':
            html = `<div class="control-group">
                    <label>锐化强度 <span class="param-value" id="ssAmtVal">1.5</span></label>
                    <input type="range" id="param_ss_amount" min="0.5" max="3.0" step="0.1" value="1.5" oninput="document.getElementById('ssAmtVal').innerText=this.value">
                </div>
                <div class="control-group">
                    <label>边界检测阈值 <span class="param-value" id="ssThreshVal">30</span></label>
                    <input type="range" id="param_ss_threshold" min="10" max="100" step="5" value="30" oninput="document.getElementById('ssThreshVal').innerText=this.value">
                </div>`;
            break;
        case 'apply_curve':
            html = `<div class="control-group">
                    <label>曲线预设</label>
                    <select id="param_curve_preset" style="width:100%; border-radius:4px; border:1px solid rgba(255,255,255,0.1); background:#2A2A2E; color:#eee; padding:5px;">
                        <option value="s_curve">S增强</option>
                        <option value="brighten_shadows">拉暗部</option>
                        <option value="compress_highlights">压高光</option>
                        <option value="invert">负片</option>
                        <option value="vintage_fade">胶片褪色</option>
                    </select>
                </div>`;
            break;
        case 'stylize':
            html = `<div class="control-group">
                    <label>风格图片 (必选)</label>
                    <input type="file" id="param_second_image" accept="image/*" style="display:block; margin-top:8px;">
                </div>
                <div class="control-group">
                    <label>优化步数 <span class="param-value" id="stStepVal">200</span></label>
                    <input type="range" id="param_st_step" min="50" max="500" step="50" value="200" oninput="document.getElementById('stStepVal').innerText=this.value">
                </div>`;
            break;
        default:
            html = `<div style="font-size:0.8rem; color:#888; padding:10px 0;">(智能处理无须多加参数设定)</div>`;
    }
    
    paramContainer.innerHTML = html;
}

/** 统一获取表单内参数的对象以供请求使用 */
function collectParams(action) {
    let params = {};
    let paramDesc = '';
    
    if (['mean_filter', 'median_filter', 'gaussian_blurring'].includes(action)) {
        const ks = parseInt(document.getElementById('param_kernel_size').value);
        params['kernel_size'] = ks;
        paramDesc = `核=${ks}`;
    } else if (action === 'apply_colormap') {
        paramDesc = `伪彩映射`;
    } else if (action === 'bilateral_filter') {
        params['kernel_size'] = parseInt(document.getElementById('param_kernel_size').value);
        params['sigma_s'] = parseFloat(document.getElementById('param_sigma_s').value);
        params['sigma_r'] = parseFloat(document.getElementById('param_sigma_r').value);
        paramDesc = `双边核=${params.kernel_size}`;
    } else if (['bandpass_filter', 'bandreject_filter'].includes(action)) {
        params['low_cut'] = parseFloat(document.getElementById('param_low_cut').value);
        params['high_cut'] = parseFloat(document.getElementById('param_high_cut').value);
        paramDesc = `频[${params.low_cut}, ${params.high_cut}]`;
    } else if (action === 'shear') {
        params['start_x_ratio'] = parseInt(document.getElementById('param_start_x').value);
        params['end_x_ratio'] = parseInt(document.getElementById('param_end_x').value);
        params['start_y_ratio'] = parseInt(document.getElementById('param_start_y').value);
        params['end_y_ratio'] = parseInt(document.getElementById('param_end_y').value);
        paramDesc = `裁切比例`;
    } else if (['false_color_channel_swap', 'synthesize_false_color_image'].includes(action)) {
        params['r_src'] = parseInt(document.getElementById('param_r_src').value);
        params['g_src'] = parseInt(document.getElementById('param_g_src').value);
        params['b_src'] = parseInt(document.getElementById('param_b_src').value);
        paramDesc = `置换 R=${params.r_src} G=${params.g_src} B=${params.b_src}`;
    } else if (action === 'add_border') {
        params['scale'] = parseFloat(document.getElementById('param_scale').value);
        paramDesc = `画框比例=${params.scale}`;
    } else if (action === 'rotate') {
        const ang = parseInt(document.getElementById('param_angle').value);
        params['angle'] = ang;
        paramDesc = `旋转=${ang}°`;
    } else if (action === 'translate') {
        const tx = parseInt(document.getElementById('param_tx').value);
        const ty = parseInt(document.getElementById('param_ty').value);
        params['tx'] = tx; params['ty'] = ty;
        paramDesc = `偏=[${tx}, ${ty}]`;
    } else if (action === 'zoom') {
        const zx = parseFloat(document.getElementById('param_z_x').value);
        const zy = parseFloat(document.getElementById('param_z_y').value);
        params['x_scale'] = zx; params['y_scale'] = zy;
        paramDesc = `缩=[${zx}, ${zy}]`;
    } else if (action === 'frosted') {
        const offset = parseInt(document.getElementById('param_offset').value);
        params['offset'] = offset;
        paramDesc = `磨砂=${offset}`;
    } else if (action === 'binarize') {
        const th = parseInt(document.getElementById('param_threshold').value);
        params['threshold'] = th;
        paramDesc = `阈=${th}`;
    } else if (action === 'adjust_brightness') {
        const beta = parseInt(document.getElementById('param_beta').value);
        params['beta'] = beta;
        paramDesc = `亮=${beta}`;
    } else if (['adjust_contrast', 'adjust_saturation'].includes(action)) {
        const alpha = parseFloat(document.getElementById('param_alpha').value);
        params['alpha'] = alpha;
        paramDesc = `度=${alpha}`;
    } else if (action === 'adjust_sharpness') {
        const amount = parseFloat(document.getElementById('param_amount').value);
        const radius = parseInt(document.getElementById('param_radius').value);
        params['amount'] = amount; params['radius'] = radius;
        paramDesc = `强=${amount.toFixed(1)}, 核=${radius}`;
    } else if (['horizontal_collage', 'vertical_collage'].includes(action)) {
        const gap = parseInt(document.getElementById('param_gap').value);
        params['gap'] = gap;
        paramDesc = `隙=${gap}px`;
    } else if (action === 'stitch_images') {
        paramDesc = `全景拼接`;
    } else if (['lowpass_filter', 'highpass_filter'].includes(action)) {
        const cutoff = parseFloat(document.getElementById('param_cutoff').value);
        params['cutoff'] = cutoff;
        paramDesc = `Cut=${cutoff}`;
    } else if (action === 'intelligent_fill_light') {
        params['strength'] = parseFloat(document.getElementById('param_ifl_strength').value);
        params['shadow_threshold'] = parseFloat(document.getElementById('param_ifl_threshold').value);
        paramDesc = `补光=${params.strength}`;
    } else if (action === 'add_noise_gaussian') {
        params['var'] = parseFloat(document.getElementById('param_var').value);
        params['mean'] = parseFloat(document.getElementById('param_mean').value);
        paramDesc = `方差=${params.var} 均值=${params.mean}`
    } else if (action === 'adjust_highlight') {
        params['strength'] = parseFloat(document.getElementById('param_ah_strength').value);
        params['highlight_threshold'] = parseFloat(document.getElementById('param_ah_threshold').value);
        paramDesc = `高光=${params.strength}`;
    } else if (['add_salt_pepper_noise_optimized', 'add_noise_salt_pepper'].includes(action)) {
        params['prob'] = parseFloat(document.getElementById('param_prob').value);
        paramDesc = `噪比=${params.prob}`;
    } else if (action === 'smart_sharpen') {
        params['amount'] = parseFloat(document.getElementById('param_ss_amount').value);
        params['edge_threshold'] = parseFloat(document.getElementById('param_ss_threshold').value);
        paramDesc = `随边锐化=${params.amount}`;
    } else if (action === 'apply_curve') {
        params['preset'] = document.getElementById('param_curve_preset').value;
        paramDesc = `曲线=${params.preset}`;
    } else if (action === 'stylize') {
        params['num_steps'] = parseInt(document.getElementById('param_st_step').value);
        paramDesc = `步数=${params.num_steps}`;
    }
    
    return { params, paramDesc };
}
