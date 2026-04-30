// 全局历史状态 (核心重构点)
let currentAction = 'mean_filter';
let videoAction = null;
let imageHistory = []; // 保存每一层状态: { img, tool, action, params, paramDesc }
let currentStep = -1;  // 当前所处的历史版本指针

// 批量处理状态记录
let isBatchMode = false;
let batchFilesData = []; // [{ name: '...', targetB64: '...' }]

const imageInput = document.getElementById('imageInput');
const folderInput = document.getElementById('folderInput');
const dropzone = document.getElementById('dropzone');
const uploadOverlay = document.getElementById('uploadOverlay');
const canvasWrapper = document.getElementById('canvasWrapper');
const mainCanvasImage = document.getElementById('mainCanvasImage');
const processBtn = document.getElementById('processBtn');
const btnUndo = document.getElementById('btnUndo');
const btnReset = document.getElementById('btnReset');
const btnExport = document.getElementById('btnExport');
const btnBatchProcess = document.getElementById('btnBatchProcess');
const historyList = document.getElementById('historyList');

// 1. 初始化拖拽事件与文件上传
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.style.borderColor = '#007acc'; });
dropzone.addEventListener('dragleave', () => { dropzone.style.borderColor = 'transparent'; });
dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.style.borderColor = 'transparent';
    if (e.dataTransfer.files.length) handleSingleFile(e.dataTransfer.files[0]);
});

uploadOverlay.addEventListener('click', () => imageInput.click());
imageInput.addEventListener('change', function(e) {
    if (e.target.files.length) handleSingleFile(e.target.files[0]);
});

// 处理导入文件夹
folderInput.addEventListener('change', function(e) {
    const files = e.target.files;
    if (files.length === 0) return;
    
    // 取出所有的图片
    const imgFiles = Array.from(files).filter(f => f.type.startsWith('image/'));
    if (imgFiles.length === 0) {
        alert("文件夹中没有图片文件！");
        return;
    }

    isBatchMode = true;
    batchFilesData = [];
    btnBatchProcess.style.display = 'inline-flex'; // 显示批量应用按钮

    // 读取第一张图作为主画板预览
    handleSingleFile(imgFiles[0]);

    // 读取所有图片保存到 batchFilesData 以备后续批量处理
    imgFiles.forEach(file => {
        const r = new FileReader();
        r.onload = ev => {
            batchFilesData.push({ name: file.name, targetB64: ev.target.result });
        };
        r.readAsDataURL(file);
    });
});

// 核心：读取图像并入栈
function handleSingleFile(file) {
    const reader = new FileReader();
    reader.onload = function(event) {
        const b64 = event.target.result;
        imageHistory = [{ img: b64, tool: '原始图', action: 'none', params: {}, paramDesc: '' }];
        currentStep = 0;
        updateWorkspace();
    };
    reader.readAsDataURL(file);
}

// 核心：根据当前状态刷新主画板、按钮、历史记录UI
function updateWorkspace() {
    if (currentStep >= 0) {
        uploadOverlay.style.display = 'none';
        canvasWrapper.style.display = 'flex'; // 使用 flex 进行内部元素居中对齐
        mainCanvasImage.src = imageHistory[currentStep].img;
        btnExport.href = imageHistory[currentStep].img; 
        
        processBtn.disabled = false;
        btnReset.disabled = currentStep === 0;
        btnUndo.disabled = currentStep === 0;
    }

    // 渲染右侧历史记录堆栈
    historyList.innerHTML = '<div class="panel-header" style="border:none; padding:0 0 10px 0; background:none;">编辑序列 / 图层</div>';
    
    // 倒序渲染历史版本
    for (let i = imageHistory.length - 1; i >= 0; i--) {
        const item = document.createElement('div');
        item.className = `history-item ${i === currentStep ? 'active' : ''}`;
        
        let toolName = imageHistory[i].tool;
        if (i !== 0) toolName += ` <span style="font-size:0.7em;opacity:0.6">${imageHistory[i].paramDesc}</span>`;

        item.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle></svg> ${toolName}`;
        
        item.addEventListener('click', () => { currentStep = i; updateWorkspace(); });
        historyList.appendChild(item);
    }
}

// 顶部栏按钮事件
btnUndo.addEventListener('click', () => { if (currentStep > 0) { currentStep--; updateWorkspace(); }});
btnReset.addEventListener('click', () => { if (currentStep > 0) { currentStep = 0; updateWorkspace(); }});

// 工具选取排他激活
const toolItems = document.querySelectorAll('.tool-item');
toolItems.forEach(item => {
    item.addEventListener('click', () => {
        toolItems.forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        currentAction = item.dataset.action;
        videoAction = item.dataset.action;
        renderParams(currentAction); // 取自 uiParamLoader.js
    });
});

// 提取 “导出流水线” 动作
function getPipelineActions() {
    let pipeline = [];
    // 提取从第 1 步到 currentStep 的动作
    for (let i = 1; i <= currentStep; i++) {
        pipeline.push({
            action: imageHistory[i].action,
            params: imageHistory[i].params
        });
    }
    return pipeline;
}

// 批量处理
btnBatchProcess.addEventListener('click', async function() {
    if (!isBatchMode || batchFilesData.length === 0) return;
    const pipeline = getPipelineActions();
    if (pipeline.length === 0) {
        alert("尚未添加任何图像处理步骤！");
        return;
    }
    
    const prevHTML = btnBatchProcess.innerHTML;
    btnBatchProcess.innerHTML = "处理中..."; 
    btnBatchProcess.disabled = true;

    try {
        const imagesB64 = batchFilesData.map(f => f.targetB64);
        const data = await processBatchAPI(imagesB64, pipeline); // 取自 api.js
        if (data.success) {
            // 下载 ZIP太复杂，直接触发多次浏览器单图下载或提示成功
            data.result_images.forEach((b64, idx) => {
                const lnk = document.createElement('a');
                lnk.download = `Batch_Processed_${batchFilesData[idx].name}`;
                lnk.href = b64;
                lnk.click();
            });
            alert("批量处理并下载完成！");
        }
    } catch(err) {
        alert("批量处理失败");
    } finally {
        btnBatchProcess.innerHTML = prevHTML;
        btnBatchProcess.disabled = false;
    }
});

// “施加效果” 按钮：单步处理主画板
processBtn.addEventListener('click', async function() {
    if (currentStep < 0 || !imageHistory[currentStep]) return;

    const { params, paramDesc } = collectParams(currentAction); // 取自 uiParamLoader.js
    const previousHTML = processBtn.innerHTML;
    processBtn.innerHTML = `处理中... <span class="spinner" style="display:inline-block; border-color:transparent; border-top-color:#fff; width:12px; height:12px; margin-left:8px;"></span>`;
    processBtn.disabled = true;

    try {
        let secondImgB64 = null;
        const fileInput = document.getElementById('param_second_image');
        if (fileInput) {
            if (fileInput.files.length === 0) {
                alert("请选择第二张参图进行处理");
                processBtn.innerHTML = previousHTML;
                processBtn.disabled = false;
                return;
            }
            const file = fileInput.files[0];
            secondImgB64 = await new Promise((resolve) => {
                const reader = new FileReader();
                reader.onload = ev => resolve(ev.target.result);
                reader.readAsDataURL(file);
            });
        }
        
        // 判断是否是连续的同一种操作，如果是，则基于"原图"（即上一层状态）进行修改替换，避免操作套娃和信息丢失
        let baseStep = currentStep;
        let isRepeatedAction = false;
        if (currentStep > 0 && imageHistory[currentStep].action === currentAction) {
            baseStep = currentStep - 1;
            isRepeatedAction = true;
        }

        const data = await processImageAPI(currentAction, imageHistory[baseStep].img, params, secondImgB64); // 取自 api.js

        if (data.success) {
            const toolName = document.querySelector(`.tool-item[data-action="${currentAction}"]`).innerText;

            const newState = {
                img: data.result_image,
                tool: toolName,
                action: currentAction,
                params: params,
                paramDesc: paramDesc
            };

            if (isRepeatedAction) {
                // 覆盖和替换掉最顶层的同类操作
                imageHistory[currentStep] = newState;
            } else {
                // 原有的逻辑：丢弃被撤销掉的废弃后续路径，追加新的操作
                imageHistory = imageHistory.slice(0, currentStep + 1);
                imageHistory.push(newState);
                currentStep = imageHistory.length - 1;
            }
            
            updateWorkspace();
        } else {
            alert("处理失败: " + data.error);
        }
    } catch (error) {
        alert("网络请求失败");
    } finally {
        processBtn.innerHTML = previousHTML;
        processBtn.disabled = false;
    }
});
