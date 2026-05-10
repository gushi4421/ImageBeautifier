/**
 * api.js
 * 负责跟后端的通信 (单步处理 / 批量流水线处理)
 */

async function processImageAPI(action, base64Image, params, ...extraBase64Images) {
    const images = [base64Image, ...extraBase64Images].filter(img => img !== null && img !== undefined);
    const response = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            action: action,
            images: images,
            params: params
        })
    });
    return await response.json();
}

async function processBatchAPI(base64Images, pipeline) {
    const response = await fetch('/api/process_pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            images: base64Images,
            pipeline: pipeline
        })
    });
    return await response.json();
}
