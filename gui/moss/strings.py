"""en / ko / ja / cn strings for the MOSS-Image GUI, over ``gui.lines.strings``.

Field labels/help: English falls back to the request field's own ``help``
metadata (``library.moss.requests``); ko / ja / cn cover the basic
(non-advanced) fields, the rest fall back to English.
"""

from __future__ import annotations

# Line-specific keys over the shared table in ``gui.lines.strings``.
UI: dict[str, dict[str, str]] = {
    "en": {
        "window_title": "MOSS-Image LoRA (alpha)",
        "model_missing": "Model not found: {path}\nRun `make download-model ARGS=moss-base` "
        "(ARGS=moss-sft for the SFT DiT), set Advanced → model_dir, or set "
        "ANIMA_MOSS_MODEL_DIR in .env.",
        "scan_stale": "{stale} captions changed since their text cache — the next "
        "preprocessing run re-encodes them.",
        "scan_dupes": "{n} images share a file name with one in another subfolder — "
        "their cache files are prefixed with the folder name.",
        "tab_i2i_preprocess": "i2i Preprocess",
        "tab_i2i_train": "i2i Train",
        "i2i_task": "Task",
        "i2i_no_tasks": "No i2i task configs under configs/moss/i2i/.",
        "i2i_guide": "An i2i LoRA learns to turn a reference image into a target: the "
        "reference rides every forward as clean latent tokens. Stage builds the "
        "pairs, Preprocess caches each target with its reference (and caption), Train "
        "fits the LoRA, Test renders new references with it. Each form starts from "
        "configs/moss/i2i/<task>/; only your changes are saved.",
        "i2i_task_colorize": "colorize: B&W manga page → colour. Staging mangafies the "
        "colour corpus; the page enters the text encoder through its vision tower, "
        "with its OCR'd lines in the caption, and as a half-size reference. Test with "
        "an empty prompt: the page alone.",
        "i2i_task_sanitize": "sanitize: removes text and speech bubbles. Staging mines "
        "in-artist near-twin pairs (texted / clean); the texted page is a full-size "
        "reference. Test with the page's caption minus its text tags.",
        "i2i_stage_group": "1 · Stage",
        "i2i_stage_hint": "Builds the task's pair tree under post_image_dataset/i2i/ "
        "from the dataset (make i2i-stage TASK={task}; knobs in the [staging] table "
        "of i2i_tasks/configs/{task}.toml). EasyControl reads the same tree.",
        "run_stage": "Stage",
        "i2i_cache_group": "2 · Preprocess",
        "i2i_train_group": "3 · Train",
        "i2i_chain": "Stage, preprocess, then train",
        "i2i_test_group": "4 · Test",
        "i2i_test_hint": "Renders each reference with the LoRA, shown beside it. "
        "ref_image is one image or a folder of images; ref_scale and ref_vision must "
        "match the cache the LoRA trained on.",
        "i2i_staged": "{n} staged pairs.",
        "i2i_not_staged": "Not staged yet: {path} is missing. Run Stage first.",
        "i2i_stage_next": "Staging done — preprocessing.",
        "i2i_need_ref": "Set ref_image to an image or a folder of images.",
        "i2i_need_prompt": "With ref_vision off, the test needs a prompt: the page's "
        "caption.",
        "test_ref": "reference",
    },
    "ko": {
        "window_title": "MOSS-Image LoRA (alpha)",
        "model_missing": "모델을 찾을 수 없습니다: {path}\n`make download-model ARGS=moss-base`"
        "를 실행하거나 (SFT DiT는 ARGS=moss-sft), 고급 → model_dir을 지정하거나, "
        ".env에 ANIMA_MOSS_MODEL_DIR을 설정하세요.",
        "scan_stale": "캡션 {stale}개가 텍스트 캐시 이후 수정되었습니다 — 다음 전처리 때 "
        "다시 인코딩됩니다.",
        "scan_dupes": "이미지 {n}장이 다른 하위 폴더의 이미지와 파일 이름이 같습니다 — "
        "해당 캐시 파일 이름에는 폴더 이름이 접두사로 붙습니다.",
        "tab_i2i_preprocess": "i2i 전처리",
        "tab_i2i_train": "i2i 학습",
        "i2i_task": "태스크",
        "i2i_no_tasks": "configs/moss/i2i/ 아래에 i2i 태스크 설정이 없습니다.",
        "i2i_guide": "i2i LoRA는 참조 이미지를 타깃 이미지로 바꾸는 법을 배웁니다: 참조 "
        "이미지는 깨끗한 잠재변수 토큰으로 매 forward에 함께 들어갑니다. 스테이징은 쌍을 "
        "만들고, 전처리는 각 타깃을 참조 이미지(와 캡션)와 함께 캐시하고, 학습은 LoRA를 "
        "학습하며, 테스트는 그 LoRA로 새 참조 이미지를 변환합니다. 각 폼은 "
        "configs/moss/i2i/<task>/에서 시작하며, 바꾼 값만 저장됩니다.",
        "i2i_task_colorize": "colorize: 흑백 만화 페이지 → 컬러. 스테이징은 컬러 코퍼스를 "
        "만화풍으로 변환합니다. 페이지는 비전 타워를 거쳐 텍스트 인코더에 들어가고 "
        "(캡션에는 OCR로 읽은 대사가 포함), 절반 크기의 참조 이미지로도 들어갑니다. "
        "테스트는 프롬프트를 비워 두세요: 페이지만 입력합니다.",
        "i2i_task_sanitize": "sanitize: 글자와 말풍선을 지웁니다. 스테이징은 같은 작가의 "
        "거의 같은 이미지 쌍(글자 있음 / 없음)을 찾습니다. 글자가 있는 페이지가 원본 크기의 "
        "참조 이미지가 됩니다. 테스트 프롬프트는 페이지의 캡션에서 글자 관련 태그를 뺀 "
        "것입니다.",
        "i2i_stage_group": "1 · 스테이징",
        "i2i_stage_hint": "데이터셋에서 post_image_dataset/i2i/ 아래에 태스크의 쌍 "
        "트리를 만듭니다 (make i2i-stage TASK={task}; 설정은 "
        "i2i_tasks/configs/{task}.toml의 [staging] 테이블). EasyControl도 같은 트리를 "
        "읽습니다.",
        "run_stage": "스테이징",
        "i2i_cache_group": "2 · 전처리",
        "i2i_train_group": "3 · 학습",
        "i2i_chain": "스테이징, 전처리 후 학습",
        "i2i_test_group": "4 · 테스트",
        "i2i_test_hint": "각 참조 이미지를 LoRA로 변환해 나란히 보여 줍니다. ref_image는 "
        "이미지 한 장 또는 이미지 폴더이며, ref_scale과 ref_vision은 LoRA를 학습한 캐시와 "
        "같아야 합니다.",
        "i2i_staged": "스테이징된 쌍 {n}개.",
        "i2i_not_staged": "아직 스테이징되지 않았습니다: {path}가 없습니다. 먼저 "
        "스테이징을 실행하세요.",
        "i2i_stage_next": "스테이징 완료 — 전처리를 시작합니다.",
        "i2i_need_ref": "ref_image를 이미지 한 장 또는 이미지 폴더로 지정하세요.",
        "i2i_need_prompt": "ref_vision이 꺼져 있으면 테스트에 프롬프트가 필요합니다: "
        "페이지의 캡션.",
        "test_ref": "참조 이미지",
    },
    "ja": {
        "window_title": "MOSS-Image LoRA（alpha）",
        "model_missing": "モデルが見つかりません: {path}\n`make download-model ARGS=moss-base`"
        " を実行する（SFT DiT は ARGS=moss-sft）か、詳細 → model_dir を指定するか、"
        ".env に ANIMA_MOSS_MODEL_DIR を設定してください。",
        "scan_stale": "{stale} 件のキャプションがテキストキャッシュ以降に変更されています — "
        "次の前処理で再エンコードされます。",
        "scan_dupes": "{n} 枚の画像が別のサブフォルダの画像と同じファイル名です — "
        "それらのキャッシュファイル名にはフォルダ名が接頭辞として付きます。",
        "tab_i2i_preprocess": "i2i 前処理",
        "tab_i2i_train": "i2i 学習",
        "i2i_task": "タスク",
        "i2i_no_tasks": "configs/moss/i2i/ に i2i タスク設定がありません。",
        "i2i_guide": "i2i LoRA は参照画像をターゲット画像に変換することを学習します: "
        "参照画像はクリーンな潜在変数トークンとして毎回の forward に加わります。"
        "ステージングがペアを作り、前処理が各ターゲットを参照画像（とキャプション）と一緒に"
        "キャッシュし、学習が LoRA を学習し、テストがその LoRA で新しい参照画像を変換します。"
        "各フォームは configs/moss/i2i/<task>/ から始まり、変更した値だけが保存されます。",
        "i2i_task_colorize": "colorize: 白黒マンガのページ → カラー。ステージングでカラー"
        "コーパスをマンガ風に変換します。ページはビジョンタワー経由でテキストエンコーダに"
        "入り（キャプションには OCR で読んだセリフを含む）、半分のサイズの参照画像としても"
        "入ります。テストではプロンプトを空にします: ページだけを入力します。",
        "i2i_task_sanitize": "sanitize: 文字と吹き出しを消します。ステージングで同じ作者の"
        "ほぼ同じ画像ペア（文字あり / なし）を探します。文字ありのページが原寸の参照画像に"
        "なります。テストのプロンプトは、そのページのキャプションから文字関連のタグを除いた"
        "ものです。",
        "i2i_stage_group": "1 · ステージング",
        "i2i_stage_hint": "データセットから post_image_dataset/i2i/ にタスクのペアツリーを"
        "作ります（make i2i-stage TASK={task}。設定は i2i_tasks/configs/{task}.toml の "
        "[staging] テーブル）。EasyControl も同じツリーを読みます。",
        "run_stage": "ステージング",
        "i2i_cache_group": "2 · 前処理",
        "i2i_train_group": "3 · 学習",
        "i2i_chain": "ステージング、前処理してから学習",
        "i2i_test_group": "4 · テスト",
        "i2i_test_hint": "各参照画像を LoRA で変換し、横に並べて表示します。ref_image は"
        "画像1枚または画像フォルダです。ref_scale と ref_vision は LoRA を学習したキャッシュと"
        "一致させてください。",
        "i2i_staged": "ステージング済みペア {n} 組。",
        "i2i_not_staged": "まだステージングされていません: {path} がありません。"
        "先にステージングを実行してください。",
        "i2i_stage_next": "ステージング完了 — 前処理を開始します。",
        "i2i_need_ref": "ref_image に画像1枚または画像フォルダを指定してください。",
        "i2i_need_prompt": "ref_vision がオフの場合、テストにはプロンプトが必要です: "
        "ページのキャプション。",
        "test_ref": "参照画像",
    },
    "cn": {
        "window_title": "MOSS-Image LoRA（alpha）",
        "model_missing": "找不到模型：{path}\n请运行 `make download-model ARGS=moss-base`"
        "（SFT 模型另需 ARGS=moss-sft），或在「高级 → model_dir」中设置，"
        "或在 .env 中设置 ANIMA_MOSS_MODEL_DIR。",
        "scan_stale": "{stale} 条标注在文本缓存之后被修改——下次预处理会自动重新编码。",
        "scan_dupes": "有 {n} 张图片与其他子文件夹中的图片同名——"
        "它们的缓存文件名会加上文件夹前缀。",
        "tab_i2i_preprocess": "图生图预处理",
        "tab_i2i_train": "图生图训练",
        "i2i_task": "任务",
        "i2i_no_tasks": "configs/moss/i2i/ 下没有图生图任务配置。",
        "i2i_guide": "图生图 LoRA 学习把参考图变成目标图：参考图以干净的潜变量 token "
        "参与每次前向。「准备配对」生成配对，「预处理」缓存每个目标及其参考图（和标注），"
        "「训练」拟合 LoRA，「测试」用它处理新的参考图。各表单从 "
        "configs/moss/i2i/<任务>/ 开始，只保存你的修改。",
        "i2i_task_colorize": "colorize：黑白漫画页 → 彩色。准备配对时把彩色语料漫画化；"
        "漫画页经视觉塔进入文本编码器（标注中带有 OCR 识别出的文字行），"
        "并作为半尺寸参考图。测试时提示词留空：只输入漫画页。",
        "i2i_task_sanitize": "sanitize：去除文字和对话框。准备配对时挖掘同一画师的"
        "近似图对（有字 / 无字）；有字的页面作为全尺寸参考图。"
        "测试时提示词用该页的标注，去掉文字相关标签。",
        "i2i_stage_group": "1 · 准备配对",
        "i2i_stage_hint": "从数据集在 post_image_dataset/i2i/ 下生成该任务的配对目录"
        "（make i2i-stage TASK={task}；参数在 i2i_tasks/configs/{task}.toml 的 "
        "[staging] 表中）。EasyControl 读取同一目录。",
        "run_stage": "准备配对",
        "i2i_cache_group": "2 · 预处理",
        "i2i_train_group": "3 · 训练",
        "i2i_chain": "准备配对、预处理后训练",
        "i2i_test_group": "4 · 测试",
        "i2i_test_hint": "用 LoRA 处理每张参考图，并与参考图并排显示。ref_image "
        "为一张图片或图片文件夹；ref_scale 和 ref_vision 须与 LoRA 训练所用的缓存一致。",
        "i2i_staged": "已准备 {n} 个配对。",
        "i2i_not_staged": "尚未准备配对：找不到 {path}。请先运行「准备配对」。",
        "i2i_stage_next": "配对准备完成——开始预处理。",
        "i2i_need_ref": "请把 ref_image 设为一张图片或图片文件夹。",
        "i2i_need_prompt": "ref_vision 关闭时，测试需要提示词：该页的标注。",
        "test_ref": "参考图",
    },
}

# Field labels + help, keyed by request field name. English is not listed:
# its label is the field name and its help the request's own metadata.
FIELDS_CN: dict[str, tuple[str, str]] = {
    "src": (
        "图片文件夹",
        "包含图片和同名 .txt 标注的文件夹，包括子文件夹"
        "（默认：Anima 的 resized 目录，其中是修订后的标注；"
        "该目录中只有标注、没有图片时，使用 image_dataset/ 中的原图）",
    ),
    "out": ("缓存文件夹", "每张图片写入两个缓存文件（文本嵌入 + 潜变量）"),
    "resolution": (
        "分辨率",
        "像素面积上限的边长——1024 表示按图片原始比例最多约 1024² 像素，"
        "两边均为 16 的倍数；不会放大",
    ),
    "caption_format": (
        "标注格式",
        "文本编码器编码的内容：layout_json = 把 Anima 标注映射为 MOSS 的 layout JSON 提示词"
        "（需要 danbooru 标签 CSV），text = 原样使用 .txt，fixed = 每张图都编码 fixed_caption。"
        "生成时请使用相同格式",
    ),
    "overwrite": ("覆盖", "即使缓存是最新的也重新编码"),
    "skip_text": ("跳过文本", "跳过文本编码器"),
    "skip_latents": ("跳过潜变量", "跳过 VAE 编码"),
    "model_dir": (
        "模型文件夹",
        "包含 MOSS-Image-Base/ 和 MOSS-Image-SFT/ 的文件夹"
        "（默认：$ANIMA_MOSS_MODEL_DIR 或 models/moss_image）",
    ),
    "cache": ("缓存文件夹", "预处理写入的缓存文件夹"),
    "output": ("输出", "LoRA 输出路径（同目录下会写入 .json 运行报告）"),
    "model": ("模型", "使用哪个 DiT（base / sft）"),
    "epochs": ("轮数", "遍历缓存的次数"),
    "max_steps": ("最大步数", "达到此步数后停止（0 = 由轮数决定）"),
    "algo": (
        "适配器",
        "lora，或 LyCORIS 的 lokr（Kronecker 积）/ loha（两组秩 r 矩阵的 Hadamard 积）",
    ),
    "rank": ("秩", "LoRA 秩（lokr：较大 Kronecker 因子的秩）"),
    "alpha": (
        "alpha",
        "LoRA alpha（lora 默认 128，lokr / loha 默认等于秩）；缩放 = alpha / 秩",
    ),
    "factor": (
        "lokr 因子",
        "仅 lokr：较小 Kronecker 因子的边长上限（-1 = 接近平方的拆分）",
    ),
    "down_init": (
        "down 初始化",
        "仅 lora：kaiming = 默认随机初始化；weight_svd = 用基础权重前 r 个右奇异向量 / √3 初始化（Anima 的 SVD-Down）",
    ),
    "channel_scaling_alpha": (
        "通道缩放 α",
        "仅 lora：按 mean|x[c]|^α 重新平衡各输入通道的 AdamW 步长（Anima 的 channel scaling）；"
        "输出不变，保存为普通 LoRA。0 = 关闭，0.5 = Anima 默认",
    ),
    "dc_isolate": (
        "DC 隔离",
        "仅 lora：把校准得到的输入直流方向（各目标的 token 均值 x̄）放进 down 的一行，"
        "其余行保持与它正交，使只有一列 up 学习与 token 无关的偏置。初始输出不变，"
        "保存为普通 LoRA。默认开启（配合 lr 5e-5）",
    ),
    "lr": ("学习率", "AdamW 学习率"),
    "warmup_ratio": ("预热比例", "线性预热占总步数的比例"),
    "grad_accum": (
        "梯度累积",
        "每次优化器更新累积的样本数；步数、轮数和损失日志仍按样本计，预热按更新计",
    ),
    "save_every_epochs": ("每 N 轮保存", "每 N 轮额外保存一次（0 = 只保存最终结果）"),
    "caption_dropout": (
        "标注丢弃率",
        "以空提示词嵌入（CFG 的无条件分支）训练一步的概率",
    ),
    "plain_caption_rate": (
        "Anima 标注比例",
        "以原始 Anima 格式标注代替其 layout JSON 训练一步的概率；"
        "仅 caption_format=layout_json 的缓存带有原始标注",
    ),
    "lora_dtype": (
        "LoRA 精度",
        "保存的 LoRA 文件精度。训练时保持 fp32 主权重，秩矩阵乘法按 bf16 进行",
    ),
    "grad_checkpointing": ("梯度检查点", "按块的激活检查点"),
    "seed": ("随机种子", "随机数种子"),
    "n": (
        "插画配对数",
        "用于训练的插画配对数（准备好的漫画化页 + resized 中的彩色目标）",
    ),
    "manga": ("漫画页数", "追加的 manga_color 漫画页数（0 = 只用插画）"),
    "holdout": ("留出插画数", "留作评估的插画配对数"),
    "manga_holdout": (
        "留出漫画作品数",
        "留作评估的漫画作品数，每部一页；其任何版本都不参与训练",
    ),
    "ref_scale": (
        "参考图缩放",
        "参考潜变量的边长 = 目标边长除以此值，取 16 的倍数（生成时用相同的值）",
    ),
    "variants": (
        "标注变体数",
        "稀疏标注数 + 1：embeds_v1 … embeds_v{variants-1} 为带 OCR 布局的颜色标注，"
        "奇数 k 去掉 tag_dropout 比例的标签，偶数 k 全部去掉（漫画页保留一个漫画标签）；"
        "至少 3，取奇数则两种各半",
    ),
    "tag_dropout": ("标签丢弃比例", "奇数 k 稀疏标注去掉的颜色标注标签比例"),
    "vision": (
        "视觉塔编码页面",
        "页面经视觉塔进入文本编码器，排在每条标注之前；关闭 = 只编码标注，不写 embeds_plain",
    ),
}

# GenerateRequest fields whose name also exists on another request with a
# different meaning.
FIELDS_CN_GENERATE: dict[str, tuple[str, str]] = {
    "prompt": ("提示词", "提示词（设置 prompts_file 或 example 时忽略）"),
    "negative_prompt": ("反向提示词", "反向提示词（管线默认为空）"),
    "prompt_format": (
        "提示词格式",
        "layout_json = 把 Anima 格式的提示词映射为 MOSS 的 layout JSON（与缓存的标注格式一致）；"
        "text = 原样使用，适用于已是 JSON 的提示词或以 text 格式缓存训练的 LoRA",
    ),
    "lora": (
        "LoRA",
        "要测试的 LoRA（留空 = 训练输出）；Base 上训练的 LoRA 也可用于 SFT",
    ),
    "lora_multiplier": ("LoRA 强度", "LoRA 强度；0.0 即基础模型"),
    "width": ("宽度", "16 的倍数"),
    "height": ("高度", "16 的倍数"),
    "steps": ("步数", "去噪步数"),
    "cfg": ("CFG", "CFG 强度；1.0 = 不做无条件前向"),
    "sampler": (
        "采样器",
        "er_sde = 同一 σ 网格上的 ER-SDE 随机采样（线条更锐利，不增加前向）；euler = 管线默认的 ODE",
    ),
    "uncond_skip": (
        "无条件分支跳过的块",
        "逗号分隔的 DiT 块编号（'' = 普通 CFG）；管线默认跳过第 22 块",
    ),
    "seed": ("随机种子", "第一个提示词的种子（每个提示词 +1）"),
    "out_dir": ("输出文件夹", "图片和 manifest.json 保存在这里"),
    "spectrum": ("Spectrum", "大部分步数预测最后一个块的输出，而不是运行全部 42 个块"),
    "dave": (
        "DAVE",
        "同提示词多样性：前几步在一组块上削弱图像 token 的逐通道均值。"
        "'' = 关闭，'auto' = 内置 MOSS 掩码，'lo-hi' = 第 lo..hi 块，或 dave_alpha.npz 路径",
    ),
    "dave_strength": ("DAVE 强度", "在池内块上去除的均值比例（0 = 关闭，1 = 全部）"),
    "ref_image": (
        "参考图",
        "图生图参考：一张图片，或图片文件夹（每张图片 × 每个提示词一个任务）。"
        "输出保持参考图比例，面积上限为 宽 × 高；LoRA 中记录的 ref_timestep / ref_isolate 会生效",
    ),
    "ref_scale": (
        "参考图缩放",
        "设置 ref_image 时：参考潜变量的边长 = 输出边长除以此值，取 16 的倍数。"
        "须与 LoRA 训练所用的缓存一致（sanitize 为 1，colorize 为 2）",
    ),
    "ref_vision": (
        "参考图进视觉塔",
        "设置 ref_image 时：参考图同时经视觉塔进入文本编码器，位于提示词之前，"
        "与 colorize 缓存的编码方式一致。此时提示词留空即只输入该页",
    ),
}

FIELDS_KO: dict[str, tuple[str, str]] = {
    "src": (
        "이미지 폴더",
        "이미지와 같은 이름의 .txt 캡션이 있는 폴더, 하위 폴더 포함"
        "(기본값: Anima의 resized 트리, 그 안의 캡션은 수정본입니다. "
        "그 트리에 캡션만 있고 이미지가 없으면 image_dataset/의 원본을 씁니다)",
    ),
    "out": (
        "캐시 폴더",
        "이미지마다 캐시 파일 두 개(텍스트 임베딩 + 잠재변수)를 씁니다",
    ),
    "resolution": (
        "해상도",
        "픽셀 면적 상한을 정사각형 변으로 — 1024는 원본 비율 그대로 최대 약 1024² 픽셀, "
        "양 변은 16의 배수. 확대는 하지 않습니다",
    ),
    "caption_format": (
        "캡션 형식",
        "텍스트 인코더가 인코딩하는 내용: layout_json = Anima 캡션을 MOSS의 layout JSON "
        "프롬프트로 변환(danbooru 태그 CSV 필요), text = .txt 그대로, fixed = 모든 이미지에 "
        "fixed_caption을 인코딩. 생성할 때도 같은 형식을 쓰세요",
    ),
    "overwrite": ("덮어쓰기", "캐시가 최신이어도 다시 인코딩"),
    "skip_text": ("텍스트 건너뛰기", "텍스트 인코더 단계를 건너뜀"),
    "skip_latents": ("잠재변수 건너뛰기", "VAE 인코딩을 건너뜀"),
    "model_dir": (
        "모델 폴더",
        "MOSS-Image-Base/와 MOSS-Image-SFT/가 있는 폴더"
        "(기본값: $ANIMA_MOSS_MODEL_DIR 또는 models/moss_image)",
    ),
    "cache": ("캐시 폴더", "전처리가 쓴 캐시 폴더"),
    "output": ("출력", "LoRA 출력 경로(같은 폴더에 .json 실행 리포트가 생깁니다)"),
    "model": ("모델", "학습할 DiT(base / sft)"),
    "epochs": ("에폭", "캐시를 반복하는 횟수"),
    "max_steps": ("최대 스텝", "이 스텝 수에서 멈춤(0 = 에폭으로 결정)"),
    "algo": (
        "어댑터",
        "lora, 또는 LyCORIS의 lokr(크로네커 곱) / loha(랭크 r 행렬 두 쌍의 아다마르 곱)",
    ),
    "rank": ("랭크", "LoRA 랭크(lokr: 큰 크로네커 인수의 랭크)"),
    "alpha": (
        "alpha",
        "LoRA alpha(lora 기본값 128, lokr / loha 기본값은 랭크와 같음); 스케일 = alpha / 랭크",
    ),
    "factor": (
        "lokr 인수",
        "lokr 전용: 작은 크로네커 인수의 변 길이 상한(-1 = 정사각형에 가까운 분할)",
    ),
    "down_init": (
        "down 초기화",
        "lora 전용: kaiming = 기본 무작위 초기화; weight_svd = 베이스 가중치의 상위 r개 "
        "오른쪽 특이벡터 / √3으로 초기화(Anima의 SVD-Down)",
    ),
    "channel_scaling_alpha": (
        "채널 스케일링 α",
        "lora 전용: 입력 채널별 AdamW 스텝을 mean|x[c]|^α로 재조정(Anima의 channel scaling). "
        "출력은 같고 일반 LoRA로 저장됩니다. 0 = 끔, 0.5 = Anima 기본값",
    ),
    "dc_isolate": (
        "DC 분리",
        "lora 전용: 보정된 입력 DC 방향(타깃별 토큰 평균 x̄)을 down의 한 행에 넣고 나머지 "
        "행은 그것과 직교하게 유지해, up의 한 열만 토큰과 무관한 바이어스를 배우게 합니다. "
        "초기 출력은 같고 일반 LoRA로 저장됩니다. 기본으로 켜짐(lr 5e-5와 함께)",
    ),
    "lr": ("학습률", "AdamW 학습률"),
    "warmup_ratio": ("워밍업 비율", "선형 워밍업이 전체 스텝에서 차지하는 비율"),
    "grad_accum": (
        "그래디언트 누적",
        "옵티마이저 업데이트 한 번에 누적하는 샘플 수. 스텝, 에폭, 손실 로그는 여전히 "
        "샘플 단위, 워밍업은 업데이트 단위로 셉니다",
    ),
    "save_every_epochs": (
        "N 에폭마다 저장",
        "N 에폭마다 추가로 저장(0 = 최종 결과만)",
    ),
    "caption_dropout": (
        "캡션 드롭아웃",
        "빈 프롬프트 임베딩(CFG의 무조건 분기)으로 한 스텝을 학습할 확률",
    ),
    "plain_caption_rate": (
        "Anima 캡션 비율",
        "layout JSON 대신 원래 Anima 형식 캡션으로 한 스텝을 학습할 확률. "
        "caption_format=layout_json 캐시에만 원래 캡션이 들어 있습니다",
    ),
    "lora_dtype": (
        "LoRA 정밀도",
        "저장되는 LoRA 파일의 정밀도. 학습 중에는 fp32 마스터 가중치를 유지하고 "
        "랭크 행렬곱은 bf16으로 수행합니다",
    ),
    "grad_checkpointing": ("그래디언트 체크포인팅", "블록 단위 활성값 체크포인팅"),
    "seed": ("시드", "난수 시드"),
    "n": (
        "일러스트 쌍 수",
        "학습에 쓸 일러스트 쌍 수(스테이징된 만화풍 페이지 + resized의 컬러 타깃)",
    ),
    "manga": ("만화 페이지 수", "추가할 manga_color 만화 페이지 수(0 = 일러스트만)"),
    "holdout": ("홀드아웃 일러스트 수", "평가용으로 남겨 두는 일러스트 쌍 수"),
    "manga_holdout": (
        "홀드아웃 만화 작품 수",
        "평가용으로 남겨 두는 만화 작품 수, 작품당 한 페이지. 그 작품의 어떤 판본도 "
        "학습에 쓰이지 않습니다",
    ),
    "ref_scale": (
        "참조 이미지 축소",
        "참조 잠재변수의 변 = 타깃 변 ÷ 이 값, 16의 배수(생성할 때도 같은 값 사용)",
    ),
    "variants": (
        "캡션 변형 수",
        "희소 캡션 수 + 1: embeds_v1 … embeds_v{variants-1}은 OCR 레이아웃이 붙은 컬러 "
        "캡션이며, 홀수 k는 태그의 tag_dropout 비율을, 짝수 k는 전부를 뺍니다(만화 "
        "페이지는 만화 태그 하나를 남김). 최소 3, 홀수면 두 종류가 반반",
    ),
    "tag_dropout": (
        "태그 드롭아웃 비율",
        "홀수 k 희소 캡션이 컬러 캡션에서 빼는 태그 비율",
    ),
    "vision": (
        "비전 타워로 페이지 인코딩",
        "페이지가 비전 타워를 거쳐 모든 캡션 앞에 텍스트 인코더로 들어갑니다. "
        "끔 = 캡션만 인코딩, embeds_plain을 쓰지 않음",
    ),
}

FIELDS_GENERATE_KO: dict[str, tuple[str, str]] = {
    "prompt": ("프롬프트", "프롬프트(prompts_file 또는 example을 지정하면 무시)"),
    "negative_prompt": (
        "네거티브 프롬프트",
        "네거티브 프롬프트(파이프라인 기본값은 빈 값)",
    ),
    "prompt_format": (
        "프롬프트 형식",
        "layout_json = Anima 형식 프롬프트를 MOSS의 layout JSON으로 변환(캐시의 캡션 "
        "형식과 동일); text = 입력 그대로, 이미 JSON인 프롬프트나 text 형식 캐시로 학습한 "
        "LoRA용",
    ),
    "lora": (
        "LoRA",
        "테스트할 LoRA(비우면 학습 출력). Base에서 학습한 LoRA는 SFT에도 쓸 수 있습니다",
    ),
    "lora_multiplier": ("LoRA 강도", "LoRA 강도. 0.0이면 베이스 모델 그대로"),
    "width": ("너비", "16의 배수"),
    "height": ("높이", "16의 배수"),
    "steps": ("스텝", "디노이징 스텝 수"),
    "cfg": ("CFG", "CFG 스케일. 1.0 = 무조건 forward 없음"),
    "sampler": (
        "샘플러",
        "er_sde = 같은 σ 그리드에서 ER-SDE 확률적 샘플링(선이 더 선명, 추가 forward 없음); "
        "euler = 파이프라인 기본 ODE",
    ),
    "uncond_skip": (
        "무조건 분기 건너뛸 블록",
        "쉼표로 구분한 DiT 블록 번호('' = 일반 CFG). 파이프라인 기본값은 22번 블록을 "
        "건너뜁니다",
    ),
    "seed": ("시드", "첫 프롬프트의 시드(프롬프트마다 +1)"),
    "out_dir": ("출력 폴더", "이미지와 manifest.json이 저장되는 곳"),
    "spectrum": (
        "Spectrum",
        "대부분의 스텝에서 42개 블록을 모두 돌리는 대신 마지막 블록의 출력을 예측",
    ),
    "dave": (
        "DAVE",
        "같은 프롬프트 다양성: 처음 몇 스텝 동안 블록 묶음에서 이미지 토큰의 채널별 평균을 "
        "약하게 합니다. '' = 끔, 'auto' = 내장 MOSS 마스크, 'lo-hi' = lo..hi번 블록, "
        "또는 dave_alpha.npz 경로",
    ),
    "dave_strength": (
        "DAVE 강도",
        "묶음 안의 블록에서 제거하는 평균의 비율(0 = 끔, 1 = 전부)",
    ),
    "ref_image": (
        "참조 이미지",
        "i2i 참조: 이미지 한 장 또는 이미지 폴더(이미지 × 프롬프트마다 작업 하나). "
        "출력은 참조 이미지 비율을 유지하고 면적 상한은 너비 × 높이. LoRA에 기록된 "
        "ref_timestep / ref_isolate가 적용됩니다",
    ),
    "ref_scale": (
        "참조 이미지 축소",
        "ref_image 지정 시: 참조 잠재변수의 변 = 출력 변 ÷ 이 값, 16의 배수. LoRA를 "
        "학습한 캐시와 같아야 합니다(sanitize 1, colorize 2)",
    ),
    "ref_vision": (
        "참조 이미지를 비전 타워로",
        "ref_image 지정 시: 참조 이미지가 비전 타워를 거쳐 프롬프트 앞에 텍스트 인코더로도 "
        "들어갑니다(colorize 캐시가 페이지를 인코딩하는 방식과 동일). 이때 프롬프트를 "
        "비우면 페이지만 입력합니다",
    ),
}

FIELDS_JA: dict[str, tuple[str, str]] = {
    "src": (
        "画像フォルダ",
        "画像と同名の .txt キャプションを含むフォルダ、サブフォルダを含む"
        "（既定: Anima の resized ツリー。そのキャプションは修正版です。"
        "そこにキャプションだけあって画像がない場合は image_dataset/ の元画像を使います）",
    ),
    "out": (
        "キャッシュフォルダ",
        "画像ごとにキャッシュファイルを2つ（テキスト埋め込み + 潜在変数）書き込みます",
    ),
    "resolution": (
        "解像度",
        "ピクセル面積の上限を正方形の辺で指定 — 1024 は元のアスペクト比のまま最大約 "
        "1024² ピクセル、両辺は 16 の倍数。拡大はしません",
    ),
    "caption_format": (
        "キャプション形式",
        "テキストエンコーダがエンコードする内容: layout_json = Anima キャプションを MOSS の "
        "layout JSON プロンプトに変換（danbooru タグ CSV が必要）、text = .txt をそのまま、"
        "fixed = すべての画像で fixed_caption をエンコード。生成時も同じ形式を使ってください",
    ),
    "overwrite": ("上書き", "キャッシュが最新でも再エンコードする"),
    "skip_text": ("テキストをスキップ", "テキストエンコーダの処理をスキップ"),
    "skip_latents": ("潜在変数をスキップ", "VAE エンコードをスキップ"),
    "model_dir": (
        "モデルフォルダ",
        "MOSS-Image-Base/ と MOSS-Image-SFT/ を含むフォルダ"
        "（既定: $ANIMA_MOSS_MODEL_DIR または models/moss_image）",
    ),
    "cache": ("キャッシュフォルダ", "前処理が書き込んだキャッシュフォルダ"),
    "output": (
        "出力",
        "LoRA の出力パス（同じフォルダに .json の実行レポートが書かれます）",
    ),
    "model": ("モデル", "学習に使う DiT（base / sft）"),
    "epochs": ("エポック数", "キャッシュを周回する回数"),
    "max_steps": ("最大ステップ数", "このステップ数で停止（0 = エポック数で決まる）"),
    "algo": (
        "アダプター",
        "lora、または LyCORIS の lokr（クロネッカー積）/ loha（ランク r 行列2組のアダマール積）",
    ),
    "rank": ("ランク", "LoRA ランク（lokr: 大きいクロネッカー因子のランク）"),
    "alpha": (
        "alpha",
        "LoRA alpha（lora の既定は 128、lokr / loha の既定はランクと同じ）。"
        "スケール = alpha / ランク",
    ),
    "factor": (
        "lokr 因子",
        "lokr のみ: 小さいクロネッカー因子の辺の上限（-1 = 正方形に近い分割）",
    ),
    "down_init": (
        "down 初期化",
        "lora のみ: kaiming = 既定のランダム初期化。weight_svd = ベース重みの上位 r 個の"
        "右特異ベクトル / √3 で初期化（Anima の SVD-Down）",
    ),
    "channel_scaling_alpha": (
        "チャネルスケーリング α",
        "lora のみ: 入力チャネルごとの AdamW のステップを mean|x[c]|^α で再調整"
        "（Anima の channel scaling）。出力は同じで、通常の LoRA として保存されます。"
        "0 = オフ、0.5 = Anima の既定",
    ),
    "dc_isolate": (
        "DC 分離",
        "lora のみ: 較正した入力 DC 方向（ターゲットごとのトークン平均 x̄）を down の1行に"
        "入れ、残りの行をそれと直交させて、up の1列だけがトークンに依存しないバイアスを"
        "学習するようにします。初期出力は同じで、通常の LoRA として保存されます。"
        "既定でオン（lr 5e-5 と組み合わせ）",
    ),
    "lr": ("学習率", "AdamW の学習率"),
    "warmup_ratio": (
        "ウォームアップ比率",
        "線形ウォームアップが全ステップに占める割合",
    ),
    "grad_accum": (
        "勾配累積",
        "オプティマイザ更新1回あたりに累積するサンプル数。ステップ数、エポック数、"
        "損失ログは引き続きサンプル単位、ウォームアップは更新単位で数えます",
    ),
    "save_every_epochs": (
        "N エポックごとに保存",
        "N エポックごとに追加で保存（0 = 最終結果のみ）",
    ),
    "caption_dropout": (
        "キャプションドロップアウト",
        "空プロンプトの埋め込み（CFG の無条件側）で1ステップ学習する確率",
    ),
    "plain_caption_rate": (
        "Anima キャプション比率",
        "layout JSON の代わりに元の Anima 形式キャプションで1ステップ学習する確率。"
        "元のキャプションを持つのは caption_format=layout_json のキャッシュだけです",
    ),
    "lora_dtype": (
        "LoRA 精度",
        "保存する LoRA ファイルの精度。学習中は fp32 のマスター重みを保持し、"
        "ランク行列の積は bf16 で計算します",
    ),
    "grad_checkpointing": (
        "勾配チェックポイント",
        "ブロック単位の活性チェックポイント",
    ),
    "seed": ("シード", "乱数シード"),
    "n": (
        "イラストペア数",
        "学習に使うイラストペア数（ステージング済みのマンガ風ページ + resized のカラーターゲット）",
    ),
    "manga": (
        "マンガページ数",
        "追加する manga_color のマンガページ数（0 = イラストのみ）",
    ),
    "holdout": ("ホールドアウトイラスト数", "評価用に取り分けるイラストペア数"),
    "manga_holdout": (
        "ホールドアウトマンガ作品数",
        "評価用に取り分けるマンガ作品数、各1ページ。その作品のどの版も学習に使われません",
    ),
    "ref_scale": (
        "参照画像の縮小",
        "参照潜在変数の辺 = ターゲットの辺 ÷ この値、16 の倍数（生成時も同じ値を使う）",
    ),
    "variants": (
        "キャプションバリアント数",
        "疎キャプション数 + 1: embeds_v1 … embeds_v{variants-1} は OCR レイアウト付きの"
        "カラーキャプションで、奇数 k はタグの tag_dropout 割合を、偶数 k は全部を除きます"
        "（マンガページはマンガタグを1つ残す）。最小 3、奇数なら2種類が半々",
    ),
    "tag_dropout": (
        "タグドロップアウト率",
        "奇数 k の疎キャプションがカラーキャプションから除くタグの割合",
    ),
    "vision": (
        "ビジョンタワーでページをエンコード",
        "ページがビジョンタワー経由で、各キャプションの前にテキストエンコーダへ入ります。"
        "オフ = キャプションのみエンコード、embeds_plain は書かない",
    ),
}

FIELDS_GENERATE_JA: dict[str, tuple[str, str]] = {
    "prompt": (
        "プロンプト",
        "プロンプト（prompts_file または example を指定すると無視）",
    ),
    "negative_prompt": (
        "ネガティブプロンプト",
        "ネガティブプロンプト（パイプラインの既定は空）",
    ),
    "prompt_format": (
        "プロンプト形式",
        "layout_json = Anima 形式のプロンプトを MOSS の layout JSON に変換（キャッシュの"
        "キャプション形式と同じ）。text = 入力どおり。すでに JSON のプロンプトや、text 形式の"
        "キャッシュで学習した LoRA 向け",
    ),
    "lora": (
        "LoRA",
        "テストする LoRA（空 = 学習出力）。Base で学習した LoRA は SFT でも使えます",
    ),
    "lora_multiplier": ("LoRA 強度", "LoRA の強度。0.0 でベースモデルそのもの"),
    "width": ("幅", "16 の倍数"),
    "height": ("高さ", "16 の倍数"),
    "steps": ("ステップ数", "デノイズのステップ数"),
    "cfg": ("CFG", "CFG スケール。1.0 = 無条件 forward なし"),
    "sampler": (
        "サンプラー",
        "er_sde = 同じ σ グリッド上の ER-SDE 確率的サンプリング（線がよりシャープ、"
        "forward の追加なし）。euler = パイプライン既定の ODE",
    ),
    "uncond_skip": (
        "無条件側でスキップするブロック",
        "カンマ区切りの DiT ブロック番号（'' = 通常の CFG）。パイプラインの既定は "
        "22 番ブロックをスキップ",
    ),
    "seed": ("シード", "最初のプロンプトのシード（プロンプトごとに +1）"),
    "out_dir": ("出力フォルダ", "画像と manifest.json の保存先"),
    "spectrum": (
        "Spectrum",
        "ほとんどのステップで 42 ブロックすべてを実行する代わりに、最後のブロックの出力を予測",
    ),
    "dave": (
        "DAVE",
        "同一プロンプトの多様性: 最初の数ステップの間、ブロック群で画像トークンのチャネルごとの"
        "平均を弱めます。'' = オフ、'auto' = 内蔵の MOSS マスク、'lo-hi' = lo..hi 番ブロック、"
        "または dave_alpha.npz のパス",
    ),
    "dave_strength": (
        "DAVE 強度",
        "群内のブロックで取り除く平均の割合（0 = オフ、1 = すべて）",
    ),
    "ref_image": (
        "参照画像",
        "i2i の参照: 画像1枚または画像フォルダ（画像 × プロンプトごとに1ジョブ）。"
        "出力は参照画像のアスペクト比を保ち、面積の上限は 幅 × 高さ。LoRA に記録された "
        "ref_timestep / ref_isolate が適用されます",
    ),
    "ref_scale": (
        "参照画像の縮小",
        "ref_image 指定時: 参照潜在変数の辺 = 出力の辺 ÷ この値、16 の倍数。LoRA を"
        "学習したキャッシュと一致させてください（sanitize は 1、colorize は 2）",
    ),
    "ref_vision": (
        "参照画像をビジョンタワーへ",
        "ref_image 指定時: 参照画像がビジョンタワー経由でプロンプトの前にテキストエンコーダ"
        "にも入ります（colorize キャッシュのページのエンコードと同じ）。このときプロンプトを"
        "空にするとページだけを入力します",
    ),
}

FIELDS: dict[str, dict[str, tuple[str, str]]] = {
    "ko": FIELDS_KO,
    "ja": FIELDS_JA,
    "cn": FIELDS_CN,
}
FIELDS_GENERATE: dict[str, dict[str, tuple[str, str]]] = {
    "ko": FIELDS_GENERATE_KO,
    "ja": FIELDS_GENERATE_JA,
    "cn": FIELDS_CN_GENERATE,
}
