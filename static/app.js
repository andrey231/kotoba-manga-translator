//═════════════════════════════════════════════════════════════════════
// i18n
//═════════════════════════════════════════════════════════════════════

const I18N = {
  en: {
    app_title:        "Kotoba",
    tab_translate:    "Translate",
    tab_editor:       "Editor",
    tab_characters:   "Characters",
    tab_glossary:     "Glossary",
    tab_settings:     "Settings",
    sett_section_detect:    "Detection",
    sett_section_render:    "Rendering",
    sett_section_translate: "Translation",
    sett_section_connection:"Connection",
    sett_section_debug:     "Diagnostics",
    sett_detect_threshold:  "Bubble detection threshold",
    sett_detect_threshold_desc: "Minimum confidence for the detector to accept a bubble. Lower → more bubbles detected (including noise). Higher → fewer, more reliable. Default: 0.50.",
    sett_sfx_threshold:     "SFX detection threshold",
    sett_sfx_threshold_desc:"Second-pass threshold for free-floating text (SFX / sound effects). Should be lower than the main threshold to catch small SFX missed in the first pass. Default: 0.30.",
    sett_min_area:          "Min bubble area (px²)",
    sett_min_area_desc:     "Bubbles smaller than this area are discarded after overlap clipping. Increase to suppress tiny false-positive detections. Default: 400.",
    sett_max_font:          "Max font size (px)",
    sett_max_font_desc:     "Upper limit for automatic font size selection. Raising this cap lets text grow larger in big bubbles. Default: 90.",
    sett_inpaint_shrink:    "Inpaint mask shrink (px)",
    sett_inpaint_shrink_desc:"Pixels to pull the inpainting mask inward from the bubble border. Prevents LaMa from accidentally erasing content outside the bubble edge. Default: 1.",
    sett_chunk_size:        "Bubbles per LLM request",
    sett_chunk_size_desc:   "How many speech bubbles are sent to the LLM in one batch. Larger batches provide more context but make JSON parsing less reliable. Default: 5.",
    sett_retries:           "LLM retries on failure",
    sett_retries_desc:      "Number of times to retry a failed LLM call (bad JSON, empty response, timeout). Higher values make translation more robust at the cost of extra time. Default: 3.",
    sett_ollama_url:        "Ollama API URL",
    sett_ollama_url_desc:   "Full URL to the Ollama generate endpoint. Change if Ollama runs on a different host or port. Default: http://localhost:11434/api/generate.",
    sett_reset:             "Reset to defaults",
    upload_drop:      "Drop chapter pages here",
    upload_or_click:  "or click to choose files (jpg, png, webp, zip, cbz)",
    upload_loading_archive: "Uploading archive: %1",
    cfg_lang:         "Translation language",
    cfg_font:         "Font path",
    cfg_debug:        "Debug boxes on result",
    cfg_debug_desc_text: "Draw colored bounding boxes on translated pages to diagnose detection and OCR results.",
    cfg_llm_debug:    "Verbose LLM logging (server console)",
    cfg_llm_debug_help: "When enabled, every prompt sent to the LLM and every response from it is printed to the server console. Useful for diagnosing translation issues. Slows things down and produces a lot of output.",
    cfg_fast_mode:    "Fast mode (no context)",
    cfg_fast_mode_help: "Skip page analysis and speaker attribution. Saves 2-3 vision-LLM calls per page (~30-60 seconds), but bubbles are translated without scene context and the character archive is not updated. Useful for quick drafts.",
    cfg_mask_debug:    "Mask debug",
    cfg_mask_debug_help: "Save text segmentation mask for each bubble as mask_debug_N.png in the results folder. Green overlay = detected text pixels.",
    legend_title:     "Box colors on result",
    legend_green:     "Green — text bubble translated successfully",
    legend_blue:      "Blue — free-floating text (SFX) translated",
    legend_orange:    "Orange — OCR found text but translation failed",
    legend_red:       "Red — OCR returned no text",
    cfg_model:        "Vision/LLM model",
    cfg_model_loading: "Loading model list...",
    cfg_model_none:   "No models found in Ollama",
    cfg_model_help:   "Multimodal Ollama models, used for character analysis, scene understanding, attribution and translation. Click 'Enter manually...' if your model is not listed.",
    job_id_label:     "Job ID:",
    job_id_what_title: "What is Job ID?",
    job_id_what_desc: "A unique identifier for the current translation job. Use it to reopen this chapter in the Editor tab later — copy the ID and paste it via the \"Load job...\" button. Results stay available on the server as long as you do not clear them manually.",
    progress_preparing: "Preparing...",
    editor_page:      "Page:",
    editor_load_job:       "Load job...",
    editor_detect_region:  "🔍 Detect region",
    editor_detecting:      "Detecting...",
    editor_detect_none:    "No bubbles detected in selected area",
    editor_detect_failed:  "Detection failed",
    editor_rerender:       "💾 Save and re-render",
    editor_font:      "Font:",
    editor_size:      "Size:",
    editor_color:     "Color:",
    editor_outline:   "Outline:",
    editor_angle:     "Angle:",
    editor_delete_bubble: "Clear translation (show original)",
    font_default:     "(default)",
    font_search:      "Search fonts...",
    editor_empty:     "Load or select a job to edit",
    editor_side_empty: "Bubbles will appear here",
    editor_label_original:   "Original",
    editor_label_translated: "Translated",
    characters_title: "Character archive",
    characters_reload: "Reload",
    characters_save:  "💾 Save",
    characters_clear: "🗑 Clear all",
    characters_clear_confirm: "Delete the ENTIRE character archive? This cannot be undone.",
    loading:          "Loading...",
    // Stages shown over processing pages
    stage_detect:     "Detecting bubbles",
    stage_ocr:        "Reading text (OCR)",
    stage_analyze:    "Analyzing scene",
    stage_attribute:  "Identifying speakers",
    stage_translate:  "Translating",
    stage_inpaint:    "Inpainting & rendering",
    stage_queue:      "In queue",
    sub_pageN:        "Page %1 of %2",
    // Log/runtime
    upload_loading:   "Uploading %1 files...",
    upload_done:      "Job created: %1, pages: %2",
    upload_error:     "Error: %1",
    ws_connected:     "WebSocket connected, starting...",
    ws_error:         "WebSocket error",
    ws_closed:        "WebSocket closed",
    starting_n:       "Starting processing of %1 pages...",
    page_done_log:    "✓ Page %1: %2 bubbles, %3s",
    done_summary:     "✓ Done: %1/%2, %3",
    done_errors:      "⚠ Failures: %1",
    issues_logged:    "Issues recorded: %1",
    elapsed_label:    "Elapsed:",
    remaining_label:  "ETA:",
    finished_label:   "Done!",
    page_of:          "Page %1 / %2",
    chars_count:      "%1 bubbles",
    pick_images:      "Please choose images",
    btn_abort:        "Abort",
    aborted_log:      "Translation aborted.",
    prompt_job_id:    "Enter job_id (visible while processing):",
    job_not_found:    "Could not load: %1",
    open_page_first:  "Open a page first",
    rerendering:      "Re-rendering...",
    rerender_failed:  "Re-render failed: %1",
    saved_n_chars:    "Saved: %1 characters",
    delete_char_q:    "Delete character %1?",
    empty_archive:    "Archive is empty. Run a translation to populate.",
    no_bubbles:       "No bubbles",
    no_ocr:           "(empty OCR)",
    export_hint:      "Download translated chapter:",
    export_zip:       "📦 Download ZIP",
    export_cbz:       "📚 Download CBZ",
    sett_glossary_desc: "Terms that will always be translated consistently. Useful for character names, place names, and special terminology.",
    gloss_col_source:  "Source (original)",
    gloss_col_target:  "Translation",
    gloss_col_note:    "Note",
    gloss_add:         "Add",
    gloss_import:      "Import JSON",
    gloss_export:      "Export JSON",
    gloss_empty:       "No entries yet. Add terms above.",
    gloss_placeholder_src:  "Original…",
    gloss_placeholder_tgt:  "Translation…",
    gloss_placeholder_note: "Note (optional)",
  },
  ru: {
    app_title:        "Kotoba",
    tab_translate:    "Перевод",
    tab_editor:       "Редактор",
    tab_characters:   "Персонажи",
    tab_glossary:     "Глоссарий",
    tab_settings:     "Настройки",
    sett_section_detect:    "Детекция",
    sett_section_render:    "Отрисовка",
    sett_section_translate: "Перевод",
    sett_section_connection:"Подключение",
    sett_section_debug:     "Диагностика",
    sett_detect_threshold:  "Порог детекции баблов",
    sett_detect_threshold_desc: "Минимальная уверенность детектора для принятия бабла. Ниже → больше баблов (включая шум). Выше → меньше, но надёжнее. Умолчание: 0.50.",
    sett_sfx_threshold:     "Порог детекции SFX",
    sett_sfx_threshold_desc:"Порог второго прохода для плавающего текста (звуковые эффекты). Должен быть ниже основного порога — ловит SFX, пропущенные в первом проходе. Умолчание: 0.30.",
    sett_min_area:          "Мин. площадь бабла (пикс²)",
    sett_min_area_desc:     "Баблы меньше этой площади удаляются после клиппинга пересечений. Увеличьте, чтобы отсеять мелкие ложные срабатывания. Умолчание: 400.",
    sett_max_font:          "Макс. размер шрифта (px)",
    sett_max_font_desc:     "Верхняя граница автоподбора размера шрифта. Увеличение позволяет тексту стать крупнее в больших баблах. Умолчание: 90.",
    sett_inpaint_shrink:    "Отступ маски инпейтинга (px)",
    sett_inpaint_shrink_desc:"Пикселей, на которые маска инпейтинга отступает от края bbox. Защищает фон рядом с баблом от случайного стирания. Умолчание: 1.",
    sett_chunk_size:        "Баблов за один LLM-запрос",
    sett_chunk_size_desc:   "Сколько баблов отправляется LLM в одном батче. Больше → больше контекста, но JSON менее надёжен. Умолчание: 5.",
    sett_retries:           "Повторных попыток LLM",
    sett_retries_desc:      "Число попыток повторить вызов при ошибке LLM (плохой JSON, пустой ответ, таймаут). Больше → стабильнее, но медленнее. Умолчание: 3.",
    sett_ollama_url:        "URL Ollama API",
    sett_ollama_url_desc:   "Полный URL эндпоинта генерации Ollama. Измените, если Ollama запущена на другом хосте или порту. Умолчание: http://localhost:11434/api/generate.",
    sett_reset:             "Сбросить настройки по умолчанию",
    upload_drop:      "Перетащите страницы главы сюда",
    upload_or_click:  "или нажмите чтобы выбрать файлы (jpg, png, webp, zip, cbz)",
    upload_loading_archive: "Загружаем архив: %1",
    cfg_lang:         "Язык перевода",
    cfg_font:         "Путь к шрифту",
    cfg_debug:        "Дебаг-рамки на результате",
    cfg_debug_desc_text: "Рисует цветные рамки на переведённых страницах для диагностики детекции и распознавания текста.",
    cfg_llm_debug:    "Подробный лог LLM (в консоли сервера)",
    cfg_llm_debug_help: "Когда включено, каждый запрос к LLM и каждый ответ от неё печатаются в консоль сервера. Полезно для диагностики проблем с переводом. Замедляет работу и производит много вывода.",
    cfg_fast_mode:    "Быстрый режим (без контекста)",
    cfg_fast_mode_help: "Пропускает анализ страницы и определение спикеров. Экономит 2-3 vision-вызова на страницу (≈30-60 секунд), но баблы переводятся без контекста сцены, и архив персонажей не обновляется. Полезно для быстрых черновых переводов.",
    cfg_mask_debug:    "Дебаг маски",
    cfg_mask_debug_help: "Сохраняет маску детекции текста для каждого бабла в папку mask_debug внутри папки results. Зелёный = обнаруженные пиксели текста.",
    legend_title:     "Цвета рамок на результате",
    legend_green:     "Зелёный — бабл переведён успешно",
    legend_blue:      "Синий — текст вне бабла (SFX) переведён",
    legend_orange:    "Оранжевый — OCR нашёл текст, но перевод не удался",
    legend_red:       "Красный — OCR не вернул текста",
    cfg_model:        "Vision/LLM модель",
    cfg_model_loading: "Загружаем список моделей...",
    cfg_model_none:   "Модели в Ollama не найдены",
    cfg_model_help:   "Мультимодальные модели Ollama для анализа персонажей, сцены, атрибуции и перевода. Если нужной модели нет в списке — нажмите «Ввести вручную...».",
    job_id_label:     "Job ID:",
    job_id_what_title: "Что такое Job ID?",
    job_id_what_desc: "Уникальный идентификатор текущего перевода. Используется чтобы потом открыть главу заново на вкладке «Редактор» — скопируйте ID и вставьте через кнопку «Загрузить джоб...». Результаты на сервере хранятся пока вы их вручную не очистите.",
    progress_preparing: "Готовится...",
    editor_page:      "Страница:",
    editor_load_job:       "Загрузить джоб...",
    editor_detect_region:  "🔍 Детектировать область",
    editor_detecting:      "Детектирование...",
    editor_detect_none:    "Баблы не найдены в выбранной области",
    editor_detect_failed:  "Ошибка детектирования",
    editor_rerender:       "💾 Сохранить и перерисовать",
    editor_font:      "Шрифт:",
    editor_size:      "Размер:",
    editor_color:     "Цвет:",
    editor_outline:   "Обводка:",
    editor_angle:     "Угол:",
    editor_delete_bubble: "Убрать перевод (показать оригинал)",
    font_default:     "(по умолчанию)",
    font_search:      "Поиск шрифта...",
    editor_empty:     "Загрузите или выберите джоб для редактирования",
    editor_side_empty: "Баблы появятся здесь",
    editor_label_original:   "Оригинал",
    editor_label_translated: "Перевод",
    characters_title: "Архив персонажей",
    characters_reload: "Обновить",
    characters_save:  "💾 Сохранить",
    characters_clear: "🗑 Очистить всё",
    characters_clear_confirm: "Удалить ВЕСЬ архив персонажей? Это действие необратимо.",
    loading:          "Загрузка...",
    stage_detect:     "Поиск баблов",
    stage_ocr:        "Распознавание текста (OCR)",
    stage_analyze:    "Анализ сцены",
    stage_attribute:  "Определение говорящих",
    stage_translate:  "Перевод",
    stage_inpaint:    "Инпейтинг и отрисовка",
    stage_queue:      "В очереди",
    sub_pageN:        "Страница %1 из %2",
    upload_loading:   "Загружаем %1 файлов...",
    upload_done:      "Job создан: %1, страниц: %2",
    upload_error:     "Ошибка: %1",
    ws_connected:     "WebSocket подключён, запускаем...",
    ws_error:         "WebSocket ошибка",
    ws_closed:        "WebSocket закрыт",
    starting_n:       "Начинаем обработку %1 страниц...",
    page_done_log:    "✓ Страница %1: %2 баблов, %3с",
    done_summary:     "✓ Готово: %1/%2, %3",
    done_errors:      "⚠ Ошибок: %1",
    issues_logged:    "Зафиксировано проблем: %1",
    elapsed_label:    "Прошло:",
    remaining_label:  "Осталось ≈",
    finished_label:   "Готово!",
    page_of:          "Страница %1 / %2",
    chars_count:      "%1 баблов",
    pick_images:      "Выберите изображения",
    btn_abort:        "Прервать",
    aborted_log:      "Перевод прерван.",
    prompt_job_id:    "Введите job_id (виден во время обработки):",
    job_not_found:    "Не удалось загрузить: %1",
    open_page_first:  "Откройте страницу",
    rerendering:      "Перерисовка...",
    rerender_failed:  "Не удалось перерисовать: %1",
    saved_n_chars:    "Сохранено: %1 персонажей",
    delete_char_q:    "Удалить персонажа %1?",
    empty_archive:    "Архив пуст. Запустите перевод, чтобы заполнить.",
    no_bubbles:       "Баблов нет",
    no_ocr:           "(пустой OCR)",
    export_hint:      "Скачать переведённую главу:",
    export_zip:       "📦 Скачать ZIP",
    export_cbz:       "📚 Скачать CBZ",
    sett_glossary_desc: "Термины, которые всегда переводятся одинаково. Полезно для имён персонажей, названий мест и специальной терминологии.",
    gloss_col_source:  "Оригинал",
    gloss_col_target:  "Перевод",
    gloss_col_note:    "Примечание",
    gloss_add:         "Добавить",
    gloss_import:      "Импорт JSON",
    gloss_export:      "Экспорт JSON",
    gloss_empty:       "Глоссарий пуст. Добавьте термины выше.",
    gloss_placeholder_src:  "Оригинал…",
    gloss_placeholder_tgt:  "Перевод…",
    gloss_placeholder_note: "Примечание (необязательно)",
  },
};

let LANG = localStorage.getItem("lang") || "en";

function t(key, ...args) {
  let s = (I18N[LANG] && I18N[LANG][key]) || I18N.en[key] || key;
  args.forEach((v, i) => { s = s.replace(`%${i + 1}`, v); });
  return s;
}

function applyI18n() {
  document.documentElement.lang = LANG;
  document.querySelectorAll("[data-i18n]").forEach(el => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  });
  document.querySelectorAll(".lang-btn").forEach(b => {
    b.classList.toggle("active", b.dataset.lang === LANG);
  });
}

document.querySelectorAll(".lang-btn").forEach(btn => {
  btn.onclick = () => {
    LANG = btn.dataset.lang;
    localStorage.setItem("lang", LANG);
    applyI18n();
    // Re-render dynamic content where it matters
    if (currentEditPage) openEditPage(currentEditPage.page);
    if (Object.keys(charactersData).length) renderCharacters();
    if ($('cfg-model').options.length) loadModels();
  };
});

//═════════════════════════════════════════════════════════════════════
// state
//═════════════════════════════════════════════════════════════════════

let currentJobId = null;
let currentJobPages = [];
let currentEditPage = null;
let deletedBubbleIdxs = new Set();
let ws = null;
let startTime = null;
let totalPagesGlobal = 0;
// Map page idx → card element while processing
const processingCards = {};

const $ = id => document.getElementById(id);

//═════════════════════════════════════════════════════════════════════
// tabs
//═════════════════════════════════════════════════════════════════════

document.querySelectorAll('.tab').forEach(tab => {
  tab.onclick = () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    tab.classList.add('active');
    $('panel-' + tab.dataset.panel).classList.add('active');
    if (tab.dataset.panel === 'characters') loadCharacters();
    if (tab.dataset.panel === 'glossary') loadGlossary();
  };
});

//═════════════════════════════════════════════════════════════════════
// upload
//═════════════════════════════════════════════════════════════════════

const uploadZone = $('upload-zone');
const fileInput = $('file-input');

uploadZone.onclick = () => fileInput.click();
['dragover', 'dragenter'].forEach(ev => uploadZone.addEventListener(ev, e => {
  e.preventDefault(); uploadZone.classList.add('drag');
}));
['dragleave', 'drop'].forEach(ev => uploadZone.addEventListener(ev, e => {
  e.preventDefault(); uploadZone.classList.remove('drag');
}));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  startUpload(Array.from(e.dataTransfer.files));
});
fileInput.onchange = () => startUpload(Array.from(fileInput.files));

async function startUpload(files) {
  const isArchive = f => /\.(zip|cbz)$/i.test(f.name);
  const archives = files.filter(isArchive);
  const images   = files.filter(f => f.type.startsWith('image/'));

  let archiveMode = false;
  if (archives.length > 0) {
    files = [archives[0]];   // only the first archive
    archiveMode = true;
  } else {
    files = images;
  }

  if (!files.length) { alert(t("pick_images")); return; }

  // Reset shared state
  $('pages-grid').innerHTML = '';
  for (const key of Object.keys(processingCards)) delete processingCards[key];
  currentJobPages = [];
  currentEditPage = null;
  $('btn-export-zip').disabled = true;
  $('btn-export-cbz').disabled = true;

  if (!archiveMode) {
    // Natural sort by filename: page2 < page10 (same logic as backend)
    files.sort((a, b) => naturalCompare(a.name, b.name));
    // Pre-create skeleton cards with blurred preview images
    const previews = await Promise.all(files.map(readFileAsDataURL));
    totalPagesGlobal = files.length;
    files.forEach((file, idx) => {
      const card = makeSkeletonCard(idx + 1, files.length, previews[idx], file.name);
      processingCards[idx + 1] = card;
      $('pages-grid').appendChild(card);
    });
  }

  const form = new FormData();
  for (const f of files) form.append('files', f);
  form.append('target_lang', $('cfg-lang').value);
  form.append('font_path', $('cfg-font').value);
  form.append('debug', $('cfg-debug').checked ? 'true' : 'false');
  form.append('llm_model', $('cfg-model').value || '');
  form.append('llm_debug', $('cfg-llm-debug').checked ? 'true' : 'false');
  form.append('fast_mode', $('cfg-fast-mode').checked ? 'true' : 'false');
  form.append('mask_debug', $('cfg-mask-debug').checked ? 'true' : 'false');
  // Settings panel values
  const sett = loadSettings();
  form.append('ollama_url',       sett.ollama_url);
  form.append('detect_threshold', sett.detect_threshold);
  form.append('sfx_threshold',    sett.sfx_threshold);
  form.append('min_bubble_area',  sett.min_bubble_area);
  form.append('max_font_size',    sett.max_font_size);
  form.append('inpaint_shrink',   sett.inpaint_shrink);
  form.append('chunk_size',       sett.chunk_size);
  form.append('translate_retries',sett.translate_retries);

  log(archiveMode ? t("upload_loading_archive", files[0].name) : t("upload_loading", files.length));
  $('progress-wrap').style.display = 'block';

  try {
    const r = await fetch('/api/upload', { method: 'POST', body: form });
    if (!r.ok) throw new Error('Upload failed');
    const data = await r.json();
    currentJobId = data.job_id;
    $('job-id-display').textContent = data.job_id;
    $('job-id-pill').classList.add('visible');
    log(t("upload_done", data.job_id, data.total_pages), 'ok');

    if (archiveMode) {
      // Page count and filenames are known only after extraction
      totalPagesGlobal = data.total_pages;
      const filenames = data.filenames || [];
      for (let i = 1; i <= data.total_pages; i++) {
        const fname = filenames[i - 1] || `page ${i}`;
        const previewUrl = `/files/uploads/${data.job_id}/${encodeURIComponent(fname)}`;
        const card = makeSkeletonCard(i, data.total_pages, previewUrl, fname);
        processingCards[i] = card;
        $('pages-grid').appendChild(card);
      }
    }

    openWebSocket(data.job_id, data.total_pages);
  } catch (e) {
    log(t("upload_error", e.message), 'err');
  }
}

function readFileAsDataURL(file) {
  return new Promise(resolve => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => resolve(null);
    r.readAsDataURL(file);
  });
}

function makeSkeletonCard(pageNum, total, previewSrc, filename) {
  const card = document.createElement('div');
  card.className = 'page-card processing';
  card.dataset.page = pageNum;

  const img = document.createElement('img');
  if (previewSrc) {
    img.src = previewSrc;
  } else {
    img.classList.add('placeholder');
  }
  card.appendChild(img);

  const overlay = document.createElement('div');
  overlay.className = 'skeleton-overlay';
  overlay.innerHTML = `
    <div class="spinner"></div>
    <div class="skeleton-stage">
      <span class="stage-label">${t("stage_queue")}</span>
      <span class="substep">${t("sub_pageN", pageNum, total)}</span>
    </div>
  `;
  card.appendChild(overlay);

  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.innerHTML = `<span>${escapeHtml(filename)}</span><span>—</span>`;
  card.appendChild(meta);

  return card;
}

function setCardStage(pageNum, stageKey) {
  const card = processingCards[pageNum];
  if (!card) return;
  const label = card.querySelector('.stage-label');
  if (label) label.textContent = t(stageKey);
}

//═════════════════════════════════════════════════════════════════════
// WebSocket progress
//═════════════════════════════════════════════════════════════════════

function openWebSocket(jobId, total) {
  $('btn-abort').style.display = 'inline-block';
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/${jobId}`);
  ws.onopen = () => {
    log(t("ws_connected"));
    ws.send(JSON.stringify({ action: 'start' }));
    startTime = Date.now();
    // First page moves from "in queue" to "detecting"
    setCardStage(1, "stage_detect");
  };
  ws.onmessage = ev => handleEvent(JSON.parse(ev.data), total);
  ws.onerror = () => log(t("ws_error"), 'err');
  ws.onclose = () => {
    $('btn-abort').style.display = 'none';
    log(t("ws_closed"));
  };
}

$('btn-abort').onclick = async () => {
  if (!currentJobId) return;
  $('btn-abort').disabled = true;
  try { await fetch(`/api/job/${currentJobId}/abort`, { method: 'POST' }); } catch (_) {}
  if (ws) { ws.close(); ws = null; }

  // Translate tab — очищаем карточки и прогресс, лог оставляем
  $('pages-grid').innerHTML = '';
  for (const key of Object.keys(processingCards)) delete processingCards[key];
  $('progress-fill').style.width = '0%';
  $('progress-text').textContent = '';
  $('progress-time').textContent = '';

  // Editor tab — сбрасываем содержимое
  $('editor-page-select').innerHTML = '';
  $('editor-image').innerHTML   = `<div class="empty">${t("editor_empty")}</div>`;
  $('editor-original').innerHTML = `<div class="empty">${t("editor_empty")}</div>`;
  $('editor-side').innerHTML    = `<div class="empty">${t("editor_side_empty")}</div>`;

  // Job pill
  $('job-id-pill').classList.remove('visible');
  $('job-id-display').textContent = '';

  currentJobPages = [];
  currentEditPage = null;
  currentJobId = null;
  $('btn-abort').disabled = false;
  $('btn-abort').style.display = 'none';
  log(t("aborted_log"), 'warn');
};

function handleEvent(evt, total) {
  if (evt.type === 'start') {
    log(t("starting_n", evt.total), 'ok');
  } else if (evt.type === 'stage') {
    // Optional stage update sent from backend, see web.py
    setCardStage(evt.page, evt.stage_key);
  } else if (evt.type === 'page_done') {
    const pct = Math.round(evt.page / total * 100);
    $('progress-fill').style.width = pct + '%';
    $('progress-text').textContent = t("page_of", evt.page, total);
    const elapsed = (Date.now() - startTime) / 1000;
    const remaining = (elapsed / evt.page) * (total - evt.page);
    $('progress-time').textContent =
      `${t("elapsed_label")} ${fmtTime(elapsed)}  •  ${t("remaining_label")} ${fmtTime(remaining)}`;
    log(t("page_done_log", evt.page, evt.bubbles.length, evt.elapsed.toFixed(1)), 'ok');

    // Cache page data locally so editor can open it without a server round-trip
    // (avoids races where the JSON file isn't yet flushed when user clicks).
    const existingIdx = currentJobPages.findIndex(p => p.page === evt.page);
    const pageRecord = {
      page: evt.page, filename: evt.filename, url: evt.url,
      elapsed: evt.elapsed, bubbles: evt.bubbles,
    };
    if (existingIdx >= 0) currentJobPages[existingIdx] = pageRecord;
    else currentJobPages.push(pageRecord);
    currentJobPages.sort((a, b) => a.page - b.page);

    replaceCardWithResult(evt);
    // Activate next card as "detecting"
    if (processingCards[evt.page + 1]) setCardStage(evt.page + 1, "stage_detect");

    // Обновляем список страниц в редакторе; при первой странице — открываем её
    populatePageSelect();
    if (!currentEditPage) openEditPage(currentJobPages[0].page);
  } else if (evt.type === 'finish') {
    $('progress-fill').style.width = '100%';
    $('progress-text').textContent = t("finished_label");
    log(t("done_summary", evt.stats.processed, evt.stats.total_pages,
          fmtTime(evt.stats.total_seconds)), 'ok');
    if (evt.stats.failed) log(t("done_errors", evt.stats.failed), 'warn');
    const issueCount = Object.values(evt.stats.errors || {}).reduce((a, b) => a + b, 0);
    if (issueCount) log(t("issues_logged", issueCount), 'warn');
    // Enable export only when there is at least one processed page
    if (evt.stats.processed > 0) {
      $('btn-export-zip').disabled = false;
      $('btn-export-cbz').disabled = false;
    }
  } else if (evt.type === 'error') {
    log(`Error: ${evt.message}`, 'err');
  }
}

function replaceCardWithResult(page) {
  const card = processingCards[page.page];
  if (!card) return;
  // Build clean result card in place
  card.classList.remove('processing');
  card.innerHTML = `
    <img src="${page.url}" alt="Page ${page.page}">
    <div class="meta">
      <span>${escapeHtml(page.filename)}</span>
      <span>${t("chars_count", page.bubbles.length)}</span>
    </div>
  `;
  card.onclick = () => openInEditor(page.page);
  delete processingCards[page.page];
}

//═════════════════════════════════════════════════════════════════════
// editor
//═════════════════════════════════════════════════════════════════════

$('btn-reload-job').onclick = async () => {
  const jobId = prompt(t("prompt_job_id"), currentJobId || '');
  if (jobId) loadJob(jobId);
};

async function loadJob(jobId) {
  try {
    const r = await fetch(`/api/job/${jobId}`);
    if (!r.ok) throw new Error('Job not found');
    currentJobId = jobId;
    $('job-id-display').textContent = jobId;
    $('job-id-pill').classList.add('visible');
    currentJobPages = await r.json();
    populatePageSelect();
    if (currentJobPages.length) openEditPage(currentJobPages[0].page);
  } catch (e) {
    alert(t("job_not_found", e.message));
  }
}

function populatePageSelect() {
  const sel = $('editor-page-select');
  sel.innerHTML = '';
  currentJobPages.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.page;
    opt.textContent = `${t("editor_page")} ${p.page} — ${p.filename}`;
    sel.appendChild(opt);
  });
  sel.onchange = () => openEditPage(parseInt(sel.value));
}

function openInEditor(pageNum) {
  if (!currentJobId) return;
  document.querySelector('.tab[data-panel="editor"]').click();
  // If we already have this page cached from page_done events, use it directly.
  // Avoids a server round-trip that might return stale or unflushed data.
  if (currentJobPages.find(p => p.page === pageNum)) {
    populatePageSelect();
    openEditPage(pageNum);
  } else {
    loadJob(currentJobId).then(() => openEditPage(pageNum));
  }
}

function openEditPage(pageNum) {
  const page = currentJobPages.find(p => p.page === pageNum);
  if (!page) return;
  currentEditPage = page;
  deletedBubbleIdxs = new Set();
  $('editor-page-select').value = pageNum;
  const origUrl = `/files/uploads/${currentJobId}/${page.filename}`;
  $('editor-original').innerHTML = `<img src="${origUrl}" alt="Original">`;
  $('editor-image').innerHTML =
    `<img src="${page.url}?t=${Date.now()}" alt="Page" id="editor-img-main">` +
    `<div class="bubble-select-overlay" id="bubble-select-overlay"></div>`;

  const side = $('editor-side');
  side.innerHTML = '';
  if (!page.bubbles.length) {
    side.innerHTML = `<div class="empty">${t("no_bubbles")}</div>`;
    return;
  }

  page.bubbles.forEach(b => {
    const item = document.createElement('div');
    item.className = 'bubble-item';
    item.dataset.bubbleIdx = b.idx;

    const fontVal = b.font_path || '';
    const sizeVal = b.font_size != null && b.font_size !== '' ? b.font_size : '';

    // text_color: [r,g,b] array or hex string → #rrggbb
    let colorVal = '#000000';
    if (Array.isArray(b.text_color)) {
      colorVal = '#' + b.text_color.map(c => c.toString(16).padStart(2, '0')).join('');
    } else if (typeof b.text_color === 'string' && b.text_color.startsWith('#')) {
      colorVal = b.text_color;
    }

    // outline_color: [r,g,b] array or null → #rrggbb
    let outlineColorVal = '#ffffff';
    if (Array.isArray(b.outline_color)) {
      outlineColorVal = '#' + b.outline_color.map(c => c.toString(16).padStart(2, '0')).join('');
    }
    const outlineWidthVal = b.outline_width || 0;

    // boolean style flags
    const isBold      = !!b.bold;
    const isItalic    = !!b.italic;
    const isUnderline = !!b.underline;
    const align       = b.text_align || 'center';

    item.innerHTML = `
      <button type="button" class="bubble-delete-btn" data-delete-idx="${b.idx}" title="${t('editor_delete_bubble')}">✕</button>
      <div class="num">#${b.idx} ${b.speaker !== 'unknown' ? '— ' + escapeHtml(b.speaker) : ''}</div>
      <div class="speaker">${b.x},${b.y} • ${b.width}×${b.height} • ${b.gender}</div>
      <div class="original">${escapeHtml(b.text) || t("no_ocr")}</div>
      <textarea data-idx="${b.idx}" data-field="translation">${escapeHtml(b.translation || '')}</textarea>
      <div class="bubble-style">
        <label data-i18n="editor_font">Font:</label>
        <span class="font-select-anchor"></span>
        <label data-i18n="editor_size">Size:</label>
        <input type="number" class="bubble-size" data-idx="${b.idx}" data-field="font_size"
               placeholder="auto" min="6" max="120" value="${sizeVal}">
        <div style="display:flex;align-items:center;gap:4px;flex-shrink:0;">
          <label data-i18n="editor_color" style="margin:0;white-space:nowrap;">Color:</label>
          <input type="color" data-idx="${b.idx}" data-field="text_color" value="${colorVal}"
                 title="${t('editor_color')}">
        </div>
      </div>
      <div class="bubble-style-row2">
        <div class="bubble-style-row2a">
          <button type="button" class="style-toggle bold-btn${isBold?' active':''}"
                  data-idx="${b.idx}" data-field="bold" title="Bold">B</button>
          <button type="button" class="style-toggle italic-btn${isItalic?' active':''}"
                  data-idx="${b.idx}" data-field="italic" title="Italic">I</button>
          <button type="button" class="style-toggle underline-btn${isUnderline?' active':''}"
                  data-idx="${b.idx}" data-field="underline" title="Underline">U</button>
          <span class="style-sep">│</span>
          <button type="button" class="style-toggle align-btn${align==='left'?' active':''}"
                  data-idx="${b.idx}" data-field="text_align" data-value="left" title="Align left">⬅</button>
          <button type="button" class="style-toggle align-btn${align==='center'?' active':''}"
                  data-idx="${b.idx}" data-field="text_align" data-value="center" title="Align center">≡</button>
          <button type="button" class="style-toggle align-btn${align==='right'?' active':''}"
                  data-idx="${b.idx}" data-field="text_align" data-value="right" title="Align right">➡</button>
        </div>
        <div class="bubble-style-row2b">
          <label style="color:#888;font-size:11px;">${t('editor_outline')}</label>
          <input type="color" data-idx="${b.idx}" data-field="outline_color"
                 value="${outlineColorVal}" title="Outline color">
          <input type="number" class="outline-width-input" data-idx="${b.idx}" data-field="outline_width"
                 min="0" max="10" value="${outlineWidthVal}" title="Outline width (px)">
          <span class="style-sep">│</span>
          <label style="color:#888;font-size:11px;">${t('editor_angle')}</label>
          <input type="number" class="outline-width-input" data-idx="${b.idx}" data-field="text_angle"
                 min="-180" max="180" step="1" value="${Math.round(b.text_angle || 0)}"
                 title="Угол поворота текста (°)" style="width:52px;">
          <label style="color:#666;font-size:11px;">°</label>
        </div>
      </div>
      <div class="bubble-style-box">
        <label>Cx:</label>
        <input type="number" data-idx="${b.idx}" data-field="box_cx" value="${Math.round(b.x + b.width/2)}" step="1" title="Центр X (пикс.)">
        <label>Cy:</label>
        <input type="number" data-idx="${b.idx}" data-field="box_cy" value="${Math.round(b.y + b.height/2)}" step="1" title="Центр Y (пикс.)">
        <label>Sw:</label>
        <input type="number" data-idx="${b.idx}" data-field="box_sw"
               value="${(b.width / (b.source_box?.[2] || b.width)).toFixed(2)}"
               data-orig-w="${b.source_box?.[2] || b.width}" min="0.05" step="0.05" title="Масштаб по ширине (1.0 = оригинал)">
        <label>Sh:</label>
        <input type="number" data-idx="${b.idx}" data-field="box_sh"
               value="${(b.height / (b.source_box?.[3] || b.height)).toFixed(2)}"
               data-orig-h="${b.source_box?.[3] || b.height}" min="0.05" step="0.05" title="Масштаб по высоте (1.0 = оригинал)">
      </div>
    `;
    // Вставляем font-select вместо placeholder
    item.querySelector('.font-select-anchor').replaceWith(makeFontSelect(b));

    side.appendChild(item);

    // Click on bubble item → highlight on image
    item.addEventListener('click', e => {
      if (e.target.closest('button,input,textarea,select')) return;
      selectBubble(item, b);
    });

    // Box inputs: focus → select; input → live overlay update
    item.querySelectorAll('.bubble-style-box input').forEach(inp => {
      inp.addEventListener('focus', () => selectBubble(item, b));
      inp.addEventListener('input', () => {
        if (item.classList.contains('selected')) refreshOverlay(b.idx);
      });
    });
  });

  // Toggle buttons: bold / italic / underline
  side.querySelectorAll('.style-toggle[data-field="bold"], .style-toggle[data-field="italic"], .style-toggle[data-field="underline"]').forEach(btn => {
    btn.onclick = () => btn.classList.toggle('active');
  });

  // Align buttons: only one active per bubble
  side.querySelectorAll('.align-btn').forEach(btn => {
    btn.onclick = () => {
      const idx = btn.dataset.idx;
      side.querySelectorAll(`.align-btn[data-idx="${idx}"]`).forEach(b2 => b2.classList.remove('active'));
      btn.classList.add('active');
    };
  });

  // Delete buttons — remove bubble item from DOM and mark for empty-translation rerender
  side.querySelectorAll('[data-delete-idx]').forEach(btn => {
    btn.onclick = () => {
      const idx = parseInt(btn.dataset.deleteIdx);
      deletedBubbleIdxs.add(idx);
      const b = currentEditPage?.bubbles.find(x => x.idx === idx);
      if (b) b.translation = '';
      btn.closest('.bubble-item')?.remove();
      // Hide overlay if the deleted bubble was selected
      const overlay = $('bubble-select-overlay');
      if (overlay) overlay.style.display = 'none';
    };
  });

  applyI18n();
}

function refreshOverlay(idx) {
  const img = $('editor-img-main');
  const overlay = $('bubble-select-overlay');
  if (!img || !overlay || !img.naturalWidth) return;
  const imgScale = img.clientWidth / img.naturalWidth;
  const sel = f => document.querySelector(`#editor-side [data-field="${f}"][data-idx="${idx}"]`);
  const getF = f => parseFloat(sel(f)?.value) || 0;
  const cx = getF('box_cx');
  const cy = getF('box_cy');
  const sw = Math.max(0.05, getF('box_sw') || 1);
  const sh = Math.max(0.05, getF('box_sh') || 1);
  // Масштаб всегда от оригинального размера, хранящегося в data-атрибуте
  const origW = parseFloat(sel('box_sw')?.dataset.origW) || 0;
  const origH = parseFloat(sel('box_sh')?.dataset.origH) || 0;
  const ew = Math.max(4, origW * sw);
  const eh = Math.max(4, origH * sh);
  overlay.style.display = 'block';
  overlay.style.left   = Math.round((cx - ew / 2) * imgScale) + 'px';
  overlay.style.top    = Math.round((cy - eh / 2) * imgScale) + 'px';
  overlay.style.width  = Math.round(ew * imgScale) + 'px';
  overlay.style.height = Math.round(eh * imgScale) + 'px';
}

function selectBubble(item, b) {
  document.querySelectorAll('#editor-side .bubble-item').forEach(el => el.classList.remove('selected'));
  item.classList.add('selected');

  const img = $('editor-img-main');
  if (!img) return;
  if (img.complete && img.naturalWidth) {
    refreshOverlay(b.idx);
  } else {
    img.onload = () => refreshOverlay(b.idx);
  }
}

//═════════════════════════════════════════════════════════════════════
// Font picker
//═════════════════════════════════════════════════════════════════════

let allFonts = [];
let fontsLoaded = false;
let fontsLoading = false;

async function loadFonts() {
  if (fontsLoaded || fontsLoading) return;
  fontsLoading = true;
  try {
    const r = await fetch('/api/fonts');
    const data = await r.json();
    allFonts = data.fonts || [];
    fontsLoaded = true;
    updateFontBtnDisplays();   // обновляем кнопки которые показывали сырой путь
  } catch (e) {
    console.warn('Could not load fonts:', e);
  } finally {
    fontsLoading = false;
  }
}

// Обновляет текст всех font-select кнопок с сырого пути на display-имя
function updateFontBtnDisplays() {
  document.querySelectorAll('.font-select-wrapper').forEach(wrapper => {
    const hidden = wrapper.querySelector('input[type="hidden"]');
    const btn    = wrapper.querySelector('.font-select-btn');
    if (!hidden || !btn) return;
    const path = hidden.value;
    if (!path) { btn.textContent = t('font_default'); return; }
    const found = allFonts.find(f => f.path === path);
    if (found) { btn.textContent = found.display; btn.title = path; }
  });
}

// Создаёт font-select в контейнере translate-вкладки (одно поле, не per-bubble)
function initGlobalFontPicker(initialPath) {
  const container = $('cfg-font-container');
  if (!container) return;

  const wrapper = document.createElement('div');
  wrapper.className = 'font-select-wrapper';
  wrapper.style.width = '100%';

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'font-select-btn';
  btn.textContent = initialPath || t('font_default');
  btn.title = initialPath || '';

  const hidden = document.createElement('input');
  hidden.type = 'hidden';
  hidden.id   = 'cfg-font';
  hidden.value = initialPath || '';

  btn.addEventListener('click', async e => {
    e.stopPropagation();
    if (activeFontBtn === btn) { closeFontDropdown(); return; }
    openFontDropdown(btn, hidden);
  });

  wrapper.append(btn, hidden);
  container.appendChild(wrapper);
}

// Single shared dropdown element (portal — appended to body to escape sidebar overflow)
let fontDropEl = null;
let activeFontBtn = null;
let activeFontHidden = null;

function _ensureFontDropdown() {
  if (fontDropEl) return;
  fontDropEl = document.createElement('div');
  fontDropEl.id = 'font-dropdown-global';
  fontDropEl.innerHTML =
    `<input class="font-dropdown-search" type="text" placeholder="">` +
    `<div class="font-dropdown-list"></div>`;
  document.body.appendChild(fontDropEl);

  fontDropEl.querySelector('.font-dropdown-search').addEventListener('input', e => {
    _fillFontList(e.target.value);
  });
  // Prevent click inside dropdown from closing it
  fontDropEl.addEventListener('mousedown', e => e.stopPropagation());
}

function _fillFontList(query) {
  if (!fontDropEl) return;
  const list = fontDropEl.querySelector('.font-dropdown-list');
  list.innerHTML = '';
  const currentPath = activeFontHidden ? activeFontHidden.value : '';

  // Default option
  const defOpt = document.createElement('div');
  defOpt.className = 'font-option font-opt-default' + (currentPath === '' ? ' selected' : '');
  defOpt.textContent = t('font_default');
  defOpt.onmousedown = e => { e.preventDefault(); _pickFont('', t('font_default')); };
  list.appendChild(defOpt);

  const q = query.trim().toLowerCase();
  const filtered = q ? allFonts.filter(f => f.display.toLowerCase().includes(q) || f.family.toLowerCase().includes(q)) : allFonts;

  filtered.forEach(font => {
    const opt = document.createElement('div');
    opt.className = 'font-option' + (font.path === currentPath ? ' selected' : '');
    opt.textContent = font.display;
    opt.title = font.path;
    opt.onmousedown = e => { e.preventDefault(); _pickFont(font.path, font.display); };
    list.appendChild(opt);
  });

  if (!filtered.length && q) {
    const empty = document.createElement('div');
    empty.className = 'font-option';
    empty.style.color = '#666';
    empty.textContent = '—';
    list.appendChild(empty);
  }
}

function _pickFont(path, display) {
  if (activeFontHidden) {
    activeFontHidden.value = path;
    activeFontHidden.dispatchEvent(new Event('change')); // для wireConfigPersistence
  }
  if (activeFontBtn) {
    activeFontBtn.textContent = display;
    activeFontBtn.title = path;
  }
  closeFontDropdown();
}

function closeFontDropdown() {
  if (fontDropEl) fontDropEl.classList.remove('open');
  activeFontBtn = null;
  activeFontHidden = null;
}

async function openFontDropdown(btn, hidden) {
  _ensureFontDropdown();
  await loadFonts();

  // Update placeholder text after i18n is applied
  fontDropEl.querySelector('.font-dropdown-search').placeholder = t('font_search');

  activeFontBtn = btn;
  activeFontHidden = hidden;

  // Position below the button (fixed, escapes sidebar overflow)
  const rect = btn.getBoundingClientRect();
  fontDropEl.style.top  = (rect.bottom + 4) + 'px';
  fontDropEl.style.left = rect.left + 'px';
  fontDropEl.style.minWidth = Math.max(rect.width, 220) + 'px';
  fontDropEl.querySelector('.font-dropdown-search').value = '';
  _fillFontList('');
  fontDropEl.classList.add('open');
  fontDropEl.querySelector('.font-dropdown-search').focus();

  // Scroll selected option into view
  requestAnimationFrame(() => {
    const sel = fontDropEl.querySelector('.font-option.selected');
    if (sel) sel.scrollIntoView({ block: 'nearest' });
  });
}

// Close on click outside
document.addEventListener('mousedown', () => closeFontDropdown());

function makeFontSelect(b) {
  const currentPath = b.font_path || '';
  const currentDisplay = currentPath
    ? (allFonts.find(f => f.path === currentPath)?.display || currentPath)
    : t('font_default');

  const wrapper = document.createElement('div');
  wrapper.className = 'font-select-wrapper';

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'font-select-btn';
  btn.textContent = currentDisplay;
  btn.title = currentPath;

  // Hidden input is what btn-rerender reads
  const hidden = document.createElement('input');
  hidden.type = 'hidden';
  hidden.dataset.idx = b.idx;
  hidden.dataset.field = 'font_path';
  hidden.value = currentPath;

  btn.addEventListener('click', async e => {
    e.stopPropagation();
    // Close if already open for this button
    if (activeFontBtn === btn) { closeFontDropdown(); return; }
    openFontDropdown(btn, hidden);
  });

  wrapper.append(btn, hidden);
  return wrapper;
}

// Pre-load fonts in background so dropdown opens instantly
loadFonts();

//═════════════════════════════════════════════════════════════════════
// Detect-region tool
//═════════════════════════════════════════════════════════════════════
let detectRegionActive = false;
let _detectDrag = null;  // {startX, startY, endX, endY, canvas, ctx}

function _getDetectCanvas() {
  let c = $('detect-sel-canvas');
  if (c) return c;
  const img = $('editor-img-main');
  if (!img) return null;
  c = document.createElement('canvas');
  c.id = 'detect-sel-canvas';
  c.width  = img.clientWidth;
  c.height = img.clientHeight;
  $('editor-image').appendChild(c);
  return c;
}

$('btn-detect-region').onclick = () => {
  detectRegionActive = !detectRegionActive;
  $('btn-detect-region').classList.toggle('tool-active', detectRegionActive);
  $('editor-image').classList.toggle('detect-mode', detectRegionActive);
  if (!detectRegionActive) {
    const c = $('detect-sel-canvas');
    if (c) c.remove();
    _detectDrag = null;
  }
};

$('editor-image').addEventListener('mousedown', e => {
  if (!detectRegionActive || !currentEditPage) return;
  const img = $('editor-img-main');
  if (!img || !img.naturalWidth) return;
  e.preventDefault();
  const rect = img.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const canvas = _getDetectCanvas();
  if (!canvas) return;
  canvas.width  = img.clientWidth;
  canvas.height = img.clientHeight;
  _detectDrag = { startX: x, startY: y, endX: x, endY: y, canvas, ctx: canvas.getContext('2d') };
});

document.addEventListener('mousemove', e => {
  if (!_detectDrag) return;
  const img = $('editor-img-main');
  if (!img) return;
  const rect = img.getBoundingClientRect();
  _detectDrag.endX = e.clientX - rect.left;
  _detectDrag.endY = e.clientY - rect.top;
  const { startX, startY, endX, endY, canvas, ctx } = _detectDrag;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const dx = endX - startX, dy = endY - startY;
  ctx.fillStyle   = 'rgba(60,200,80,0.10)';
  ctx.strokeStyle = '#3fc85a';
  ctx.lineWidth   = 2;
  ctx.setLineDash([5, 4]);
  ctx.fillRect(startX, startY, dx, dy);
  ctx.strokeRect(startX, startY, dx, dy);
});

document.addEventListener('mouseup', async e => {
  if (!_detectDrag || !currentEditPage) return;
  const { startX, startY, endX, endY, canvas } = _detectDrag;
  _detectDrag = null;
  // Clear canvas
  canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);

  const img = $('editor-img-main');
  if (!img || !img.naturalWidth) return;

  // Map CSS-px → image-px
  const scale = img.naturalWidth / img.clientWidth;
  const x = Math.round(Math.min(startX, endX) * scale);
  const y = Math.round(Math.min(startY, endY) * scale);
  const w = Math.round(Math.abs(endX - startX) * scale);
  const h = Math.round(Math.abs(endY - startY) * scale);
  if (w < 10 || h < 10) return;

  const btn = $('btn-detect-region');
  btn.disabled = true;
  btn.textContent = t('editor_detecting');

  try {
    const r = await fetch(
      `/api/job/${currentJobId}/page/${currentEditPage.page}/detect-region`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x, y, w, h }),
      }
    );
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);

    if (!data.new_bubbles || data.new_bubbles.length === 0) {
      alert(t('editor_detect_none'));
      return;
    }

    // Merge new bubbles + update URL in client-side page data
    currentEditPage.bubbles.push(...data.new_bubbles);
    if (data.url) currentEditPage.url = data.url;

    // Rebuild page (image + sidebar)
    openEditPage(currentEditPage.page);

    // Re-enable detect mode on the fresh image
    if (detectRegionActive) {
      $('editor-image').classList.add('detect-mode');
      _getDetectCanvas();
    }

  } catch (ex) {
    alert(t('editor_detect_failed') + ': ' + ex.message);
  } finally {
    btn.disabled = false;
    btn.textContent = t('editor_detect_region');
    btn.classList.toggle('tool-active', detectRegionActive);
  }
});

$('btn-rerender').onclick = async () => {
  if (!currentEditPage) { alert(t("open_page_first")); return; }
  // Собираем updates по idx — все поля каждого бабла
  const byIdx = {};
  document.querySelectorAll('#editor-side [data-idx]').forEach(el => {
    const idx = parseInt(el.dataset.idx);
    if (!byIdx[idx]) byIdx[idx] = { idx };
    const field = el.dataset.field;
    if (!field) return;
    if (field === 'font_size') {
      const v = el.value.trim();
      byIdx[idx][field] = v === '' ? null : parseInt(v);
    } else if (field === 'font_path') {
      byIdx[idx][field] = el.value.trim() || null;
    } else if (field === 'bold' || field === 'italic' || field === 'underline') {
      byIdx[idx][field] = el.classList.contains('active');
    } else if (field === 'text_align') {
      if (el.classList.contains('active') && el.dataset.value) {
        byIdx[idx]['text_align'] = el.dataset.value;
      }
    } else if (field === 'outline_width') {
      byIdx[idx][field] = parseInt(el.value) || 0;
    } else if (field === 'text_angle') {
      byIdx[idx][field] = parseFloat(el.value) || 0;
    } else if (field === 'box_cx' || field === 'box_cy') {
      byIdx[idx][field] = parseFloat(el.value) || 0;
    } else if (field === 'box_sw' || field === 'box_sh') {
      byIdx[idx][field] = Math.max(0.05, parseFloat(el.value) || 1.0);
    } else {
      byIdx[idx][field] = el.value;
    }
  });
  // Default text_align = center for bubbles where no align button was active
  const updates = Object.values(byIdx);
  updates.forEach(u => { if (!u.text_align) u.text_align = 'center'; });
  // Включаем удалённые баблы с пустым переводом, чтобы сервер не рисовал их
  deletedBubbleIdxs.forEach(idx => {
    if (!byIdx[idx]) updates.push({ idx, translation: '' });
    else byIdx[idx].translation = '';
  });
  const btn = $('btn-rerender');
  btn.disabled = true; btn.textContent = t("rerendering");
  try {
    const r = await fetch(
      `/api/job/${currentJobId}/page/${currentEditPage.page}/render`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bubbles: updates }),
      }
    );
    if (!r.ok) throw new Error('render failed');
    const data = await r.json();
    currentEditPage.bubbles = data.bubbles;
    $('editor-image').innerHTML =
      `<img src="${data.url}?t=${Date.now()}" alt="Page" id="editor-img-main">` +
      `<div class="bubble-select-overlay" id="bubble-select-overlay"></div>`;
  } catch (e) {
    alert(t("rerender_failed", e.message));
  } finally {
    btn.disabled = false; btn.textContent = t("editor_rerender");
  }
};

// Arrow-key navigation in the Editor tab
document.addEventListener('keydown', e => {
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  // Ignore when focus is inside a text field / textarea
  const tag = (document.activeElement || {}).tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  // Ignore when editor panel is not active
  if (!document.getElementById('panel-editor').classList.contains('active')) return;
  if (!currentJobPages.length || !currentEditPage) return;
  const idx = currentJobPages.findIndex(p => p.page === currentEditPage.page);
  if (idx === -1) return;
  const next = e.key === 'ArrowRight' ? idx + 1 : idx - 1;
  if (next < 0 || next >= currentJobPages.length) return;
  e.preventDefault();
  openEditPage(currentJobPages[next].page);
});

//═════════════════════════════════════════════════════════════════════
// characters
//═════════════════════════════════════════════════════════════════════

let charactersData = {};

$('btn-reload-chars').onclick = loadCharacters;
$('btn-save-chars').onclick = saveCharacters;
$('btn-clear-chars').onclick = async () => {
  if (!Object.keys(charactersData).length) return;
  if (!confirm(t("characters_clear_confirm"))) return;
  await fetch('/api/characters', { method: 'DELETE' });
  charactersData = {};
  renderCharacters();
};

async function loadCharacters() {
  const r = await fetch('/api/characters');
  charactersData = await r.json();
  renderCharacters();
}

function renderCharacters() {
  const list = $('characters-list');
  list.innerHTML = '';
  const ids = Object.keys(charactersData);
  if (!ids.length) {
    list.innerHTML = `<div class="empty">${t("empty_archive")}</div>`;
    return;
  }
  ids.forEach(cid => {
    const c = charactersData[cid];
    const card = document.createElement('div');
    card.className = 'character-card';
    const firstSeenLabel = LANG === 'ru' ? 'впервые на стр.' : 'first seen p.';
    const labelName = LANG === 'ru' ? 'Имя' : 'Name';
    const labelGender = LANG === 'ru' ? 'Пол' : 'Gender';
    const labelAppearance = LANG === 'ru' ? 'Внешность' : 'Appearance';
    const labelNotes = LANG === 'ru' ? 'Заметки' : 'Notes';
    const btnDelete = LANG === 'ru' ? 'Удалить' : 'Delete';
    card.innerHTML = `
      <div class="id-tag">id: ${cid} • ${firstSeenLabel} ${c.first_seen || '?'}</div>
      <div class="field">
        <label>${labelName}</label>
        <input data-id="${cid}" data-field="name" value="${escapeHtml(c.name || '')}">
      </div>
      <div class="field">
        <label>${labelGender}</label>
        <select data-id="${cid}" data-field="gender">
          <option ${c.gender === 'male' ? 'selected' : ''}>male</option>
          <option ${c.gender === 'female' ? 'selected' : ''}>female</option>
          <option ${c.gender === 'unknown' ? 'selected' : ''}>unknown</option>
        </select>
      </div>
      <div class="field">
        <label>${labelAppearance}</label>
        <textarea data-id="${cid}" data-field="appearance">${escapeHtml(c.appearance || '')}</textarea>
      </div>
      <button class="btn-danger" data-del="${cid}" style="align-self:center;">${btnDelete}</button>
      <div class="field" style="grid-column: 1 / -1;">
        <label>${labelNotes}</label>
        <textarea data-id="${cid}" data-field="notes">${escapeHtml(c.notes || '')}</textarea>
      </div>
    `;
    list.appendChild(card);
  });

  list.querySelectorAll('[data-id]').forEach(el => {
    el.onchange = () => {
      const id = el.dataset.id, field = el.dataset.field;
      if (charactersData[id]) charactersData[id][field] = el.value;
    };
  });
  list.querySelectorAll('[data-del]').forEach(btn => {
    btn.onclick = async () => {
      const id = btn.dataset.del;
      if (!confirm(t("delete_char_q", id))) return;
      await fetch(`/api/characters/${id}`, { method: 'DELETE' });
      delete charactersData[id];
      renderCharacters();
    };
  });
}

async function saveCharacters() {
  const r = await fetch('/api/characters', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(charactersData),
  });
  const data = await r.json();
  alert(t("saved_n_chars", data.count));
}

//═════════════════════════════════════════════════════════════════════
// utils
//═════════════════════════════════════════════════════════════════════

function log(msg, cls = '') {
  const el = document.createElement('div');
  if (cls) el.className = cls;
  el.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  $('log').appendChild(el);
  $('log').scrollTop = $('log').scrollHeight;
}

function fmtTime(s) {
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60), sec = Math.round(s % 60);
  return `${m}m ${sec}s`;
}

function naturalCompare(a, b) {
  // Splits by runs of digits and compares them numerically:
  // page2.png < page10.png; chapter_01 < chapter_2 (since 1 < 2)
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
}

function escapeHtml(s) {
  return String(s || '').replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])
  );
}

//═════════════════════════════════════════════════════════════════════
// Settings panel
//═════════════════════════════════════════════════════════════════════

const SETTINGS_DEFAULTS = {
  detect_threshold:  0.50,
  sfx_threshold:     0.30,
  min_bubble_area:   400,
  max_font_size:     90,
  inpaint_shrink:    1,
  chunk_size:        5,
  translate_retries: 3,
  ollama_url:        'http://localhost:11434/api/generate',
};

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem('kotoba_settings') || '{}');
    return { ...SETTINGS_DEFAULTS, ...saved };
  } catch { return { ...SETTINGS_DEFAULTS }; }
}

function applySettingsToUI(sett) {
  $('sett-detect-threshold').value = sett.detect_threshold;
  $('sett-sfx-threshold').value    = sett.sfx_threshold;
  $('sett-min-area').value         = sett.min_bubble_area;
  $('sett-max-font').value         = sett.max_font_size;
  $('sett-inpaint-shrink').value   = sett.inpaint_shrink;
  $('sett-chunk-size').value       = sett.chunk_size;
  $('sett-retries').value          = sett.translate_retries;
  $('sett-ollama-url').value       = sett.ollama_url;
}

function saveSettings() {
  const sett = {
    detect_threshold:  parseFloat($('sett-detect-threshold').value) || SETTINGS_DEFAULTS.detect_threshold,
    sfx_threshold:     parseFloat($('sett-sfx-threshold').value)    || SETTINGS_DEFAULTS.sfx_threshold,
    min_bubble_area:   parseInt($('sett-min-area').value)           || SETTINGS_DEFAULTS.min_bubble_area,
    max_font_size:     parseInt($('sett-max-font').value)           || SETTINGS_DEFAULTS.max_font_size,
    inpaint_shrink:    parseInt($('sett-inpaint-shrink').value),
    chunk_size:        parseInt($('sett-chunk-size').value)         || SETTINGS_DEFAULTS.chunk_size,
    translate_retries: parseInt($('sett-retries').value)            || SETTINGS_DEFAULTS.translate_retries,
    ollama_url:        $('sett-ollama-url').value.trim()            || SETTINGS_DEFAULTS.ollama_url,
  };
  localStorage.setItem('kotoba_settings', JSON.stringify(sett));
}

// Wire settings controls → auto-save on change
['sett-detect-threshold','sett-sfx-threshold','sett-min-area','sett-max-font',
 'sett-inpaint-shrink','sett-chunk-size','sett-retries','sett-ollama-url'
].forEach(id => {
  const el = $(id);
  if (el) el.addEventListener('change', saveSettings);
});

$('btn-reset-settings').addEventListener('click', () => {
  applySettingsToUI(SETTINGS_DEFAULTS);
  localStorage.removeItem('kotoba_settings');
  $('cfg-debug').checked     = false;
  $('cfg-llm-debug').checked = false;
  $('cfg-mask-debug').checked = false;
  saveConfig();
});

// ─── Glossary ────────────────────────────────────────────────────────────────

let _glossary = [];

function _esc(s) {
  return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

async function loadGlossary() {
  try {
    const r = await fetch('/api/glossary');
    _glossary = await r.json();
  } catch(e) { _glossary = []; }
  renderGlossaryTable();
}

function renderGlossaryTable() {
  const tbody = $('glossary-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (_glossary.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" class="glossary-empty">${t('gloss_empty')}</td></tr>`;
    return;
  }

  _glossary.forEach((entry, idx) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><input class="gloss-cell" data-field="source" value="${_esc(entry.source)}"></td>
      <td><input class="gloss-cell" data-field="target" value="${_esc(entry.target)}"></td>
      <td><input class="gloss-cell" data-field="note"   value="${_esc(entry.note||'')}"></td>
      <td><button class="gloss-del" title="Delete">×</button></td>
    `;
    tbody.appendChild(tr);

    tr.querySelectorAll('.gloss-cell').forEach(inp => {
      inp.addEventListener('change', async () => {
        _glossary[idx][inp.dataset.field] = inp.value;
        await fetch(`/api/glossary/${idx}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(_glossary[idx]),
        });
      });
    });

    tr.querySelector('.gloss-del').addEventListener('click', async () => {
      await fetch(`/api/glossary/${idx}`, { method: 'DELETE' });
      _glossary.splice(idx, 1);
      renderGlossaryTable();
    });
  });
}

$('btn-gloss-add').addEventListener('click', async () => {
  const src  = $('gloss-new-src').value.trim();
  const tgt  = $('gloss-new-tgt').value.trim();
  const note = $('gloss-new-note').value.trim();
  if (!src || !tgt) return;
  await fetch('/api/glossary', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source: src, target: tgt, note }),
  });
  $('gloss-new-src').value = '';
  $('gloss-new-tgt').value = '';
  $('gloss-new-note').value = '';
  $('gloss-new-src').focus();
  await loadGlossary();
});

['gloss-new-src', 'gloss-new-tgt', 'gloss-new-note'].forEach(id => {
  $(id).addEventListener('keydown', e => { if (e.key === 'Enter') $('btn-gloss-add').click(); });
});

$('btn-gloss-export').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(_glossary, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'kotoba_glossary.json';
  a.click();
});

$('btn-gloss-import').addEventListener('click', () => $('gloss-import-file').click());

$('gloss-import-file').addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  let entries;
  try { entries = JSON.parse(await file.text()); } catch { alert('Invalid JSON'); return; }
  if (!Array.isArray(entries)) { alert('Expected a JSON array'); return; }
  for (const en of entries) {
    if (en.source && en.target) {
      await fetch('/api/glossary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(en),
      });
    }
  }
  e.target.value = '';
  await loadGlossary();
});

// Initial render
applyI18n();
// Font picker создаём ДО restoreConfig, чтобы $('cfg-font') уже существовал
initGlobalFontPicker('arial.ttf');
loadModels();
restoreConfig();
wireConfigPersistence();
applySettingsToUI(loadSettings());

// ─── persisted config (lang/font/model) ─────────────────────────────────────

function restoreConfig() {
  // Language (translation target), font path — restore on load.
  // Model is restored inside loadModels() once the list arrives.
  const saved = JSON.parse(localStorage.getItem("config") || "{}");
  if (saved.target_lang && hasOption("cfg-lang", saved.target_lang)) {
    $('cfg-lang').value = saved.target_lang;
  }
  if (saved.font_path) {
    const fi = $('cfg-font');
    if (fi) {
      fi.value = saved.font_path;
      // Обновляем текст кнопки; display-имя придёт после loadFonts
      const btn = document.querySelector('#cfg-font-container .font-select-btn');
      if (btn) { btn.textContent = saved.font_path; btn.title = saved.font_path; }
    }
  }
  if (saved.debug) {
    $('cfg-debug').checked = !!saved.debug;
  }
  if (saved.llm_debug) {
    $('cfg-llm-debug').checked = !!saved.llm_debug;
  }
  if (saved.fast_mode) {
    $('cfg-fast-mode').checked = !!saved.fast_mode;
  }
  if (saved.mask_debug) {
    $('cfg-mask-debug').checked = !!saved.mask_debug;
  }
}

function saveConfig() {
  const cfg = {
    target_lang: $('cfg-lang').value,
    font_path: $('cfg-font').value,
    debug: $('cfg-debug').checked,
    llm_debug: $('cfg-llm-debug').checked,
    fast_mode: $('cfg-fast-mode').checked,
    mask_debug: $('cfg-mask-debug').checked,
    llm_model: $('cfg-model').value || "",
  };
  localStorage.setItem("config", JSON.stringify(cfg));
}

function wireConfigPersistence() {
  // Persist any change to any config control
  ['cfg-lang', 'cfg-font', 'cfg-debug', 'cfg-llm-debug', 'cfg-fast-mode', 'cfg-mask-debug', 'cfg-model'].forEach(id => {
    const el = $(id);
    if (!el) return;
    el.addEventListener('change', saveConfig);
    el.addEventListener('input', saveConfig);
  });
}

function hasOption(selectId, value) {
  const sel = $(selectId);
  if (!sel) return false;
  return Array.from(sel.options).some(o => o.value === value);
}

$('btn-export-zip').onclick = () => exportArchive('zip');
$('btn-export-cbz').onclick = () => exportArchive('cbz');

function exportArchive(fmt) {
  if (!currentJobId) return;
  // Anchor-based download: browser handles the file save dialog and
  // sends Content-Disposition headers from server.
  const url = `/api/job/${currentJobId}/export?fmt=${fmt}`;
  const a = document.createElement('a');
  a.href = url;
  a.download = '';   // honors server-side Content-Disposition
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

async function loadModels() {
  const sel = $('cfg-model');
  try {
    const r = await fetch('/api/models');
    const data = await r.json();
    sel.innerHTML = '';

    // Backend error talking to Ollama
    if (data.error) {
      console.warn('Ollama error:', data.error);
      log(`Ollama: ${data.error}`, 'warn');
      const opt = document.createElement('option');
      opt.textContent = t('cfg_model_none');
      opt.disabled = true;
      sel.appendChild(opt);
      // Also expose a manual-input option so user can still proceed
      addManualEntryOption(sel);
      return;
    }

    const multimodal = data.models.filter(m => m.multimodal && !m.ocr);
    const others = data.models.filter(m => !m.multimodal && !m.ocr);
    const ocrs = data.models.filter(m => m.ocr);

    if (!data.models.length) {
      console.warn('Ollama has no models. Host:', data.ollama_host);
      log(`No models in Ollama at ${data.ollama_host}. Run "ollama pull gemma3:27b"`, 'warn');
      const opt = document.createElement('option');
      opt.textContent = t('cfg_model_none');
      opt.disabled = true;
      sel.appendChild(opt);
      addManualEntryOption(sel);
      return;
    }

    console.log(`Loaded ${data.models.length} model(s) from ${data.ollama_host}`,
                 'multimodal:', multimodal.map(m => m.name),
                 'other:', others.map(m => m.name),
                 'ocr:', ocrs.map(m => m.name));

    const appendGroup = (label, list) => {
      if (!list.length) return;
      const group = document.createElement('optgroup');
      group.label = label;
      list.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = `${m.name}${m.size_gb ? ` (${m.size_gb} GB)` : ''}`;
        if (m.name === data.default) opt.selected = true;
        group.appendChild(opt);
      });
      sel.appendChild(group);
    };

    appendGroup(
      LANG === 'ru' ? 'Мультимодальные' : 'Multimodal',
      multimodal
    );
    appendGroup(
      LANG === 'ru' ? 'Прочие (могут не поддерживать изображения)' : 'Other (may not support images)',
      others
    );
    appendGroup(
      LANG === 'ru' ? 'OCR (не подходят для перевода)' : 'OCR (not suitable for translation)',
      ocrs
    );
    addManualEntryOption(sel);

    // Restore saved model if present in the list
    const saved = JSON.parse(localStorage.getItem("config") || "{}");
    if (saved.llm_model && hasOption("cfg-model", saved.llm_model)) {
      sel.value = saved.llm_model;
    }
  } catch (e) {
    console.error('loadModels failed:', e);
    sel.innerHTML = `<option disabled>${t('cfg_model_none')}</option>`;
    addManualEntryOption(sel);
  }
}

function addManualEntryOption(sel) {
  // Lets the user type a model name manually if dropdown isn't enough
  const sep = document.createElement('option');
  sep.disabled = true; sep.textContent = '──────────';
  sel.appendChild(sep);
  const manual = document.createElement('option');
  manual.value = '__manual__';
  manual.textContent = LANG === 'ru' ? '✎ Ввести вручную...' : '✎ Enter manually...';
  sel.appendChild(manual);

  sel.onchange = () => {
    if (sel.value === '__manual__') {
      const name = prompt(LANG === 'ru' ? 'Введите имя модели Ollama:' : 'Enter Ollama model name:', '');
      if (name) {
        // Add the typed name as a new option and select it
        const custom = document.createElement('option');
        custom.value = name; custom.textContent = name + ' (custom)';
        custom.selected = true;
        sel.insertBefore(custom, sep);
      } else {
        sel.selectedIndex = 0;
      }
    }
  };
}
