(() => {
  'use strict';

  const state = {
    bootstrap: {},
    aliases: {},
    vocabulary: new Map(),
    blockedTags: new Set(),
    videos: [],
    currentVideo: null,
    videoObjectUrl: null,
    references: [],
    selectedReferenceId: null,
    wanted: [],
    avoided: [],
    intent: 'wanted',
    scope: 'subject',
    searchTimer: null,
    searchToken: 0,
    cropPointer: null,
    job: null,
    pollTimer: null,
    candidates: [],
    selectedIds: new Set(),
    activeCandidateIndex: -1,
    noticeTimer: null,
    videoRequestToken: 0,
    videoLoading: false,
    captureLoading: false,
    pollInFlight: false,
    logoProfiles: [],
    logoProfileRoot: '',
    logoModel: { ready: false, details: '' },
    logoDefaultEnabled: true,
    logoDefaultLoaded: false,
    logoProfilesLoaded: false,
    logoChoiceTouched: false,
    logoSelectedProfileId: '',
    logoProfilesToken: 0,
    logoPreviewToken: 0,
    logoPreviewContext: null,
    logoPreviewData: null,
  };

  const $ = (id) => document.getElementById(id);
  const els = {
    globalNotice: $('globalNotice'), modelStatus: $('modelStatus'), shortcutButton: $('shortcutButton'),
    videoSelect: $('videoSelect'), videoFile: $('videoFile'), videoMeta: $('videoMeta'), videoPreview: $('videoPreview'),
    referenceFiles: $('referenceFiles'), referenceTime: $('referenceTime'), captureReference: $('captureReference'),
    referenceList: $('referenceList'), referenceCount: $('referenceCount'),
    cropEmpty: $('cropEmpty'), cropStage: $('cropStage'), referenceCanvas: $('referenceCanvas'), cropReadout: $('cropReadout'),
    resetCrop: $('resetCrop'), selectedReferenceName: $('selectedReferenceName'),
    wantedTab: $('wantedTab'), avoidedTab: $('avoidedTab'), tagSearch: $('tagSearch'), tagSearchSpinner: $('tagSearchSpinner'),
    scopeButtons: [...document.querySelectorAll('.scope-button')], tagSuggestions: $('tagSuggestions'), tagWarning: $('tagWarning'),
    wantedTags: $('wantedTags'), avoidedTags: $('avoidedTags'), wantedCount: $('wantedCount'), avoidedCount: $('avoidedCount'), bucketTitle: $('bucketTitle'),
    summaryVideo: $('summaryVideo'), summaryReference: $('summaryReference'), summaryTags: $('summaryTags'), readyBadge: $('readyBadge'), analyzeButton: $('analyzeButton'),
    logoRemovalToggle: $('logoRemovalToggle'), logoProfileSelect: $('logoProfileSelect'), logoProfileStatus: $('logoProfileStatus'), logoProfileEmpty: $('logoProfileEmpty'), logoModelBadge: $('logoModelBadge'),
    startTime: $('startTime'), endTime: $('endTime'), coarseStep: $('coarseStep'), denseStep: $('denseStep'), candidateCount: $('candidateCount'),
    identityThreshold: $('identityThreshold'), tagThreshold: $('tagThreshold'), qualityMin: $('qualityMin'), rescue: $('rescue'),
    identityWeight: $('identityWeight'), conditionWeight: $('conditionWeight'), qualityWeight: $('qualityWeight'), weightTotal: $('weightTotal'),
    resultsPanel: $('resultsPanel'), progressArea: $('progressArea'), progressMessage: $('progressMessage'), progressPercent: $('progressPercent'), progressBar: $('progressBar'), cancelJob: $('cancelJob'),
    shortageNotice: $('shortageNotice'), resultsEmpty: $('resultsEmpty'), resultsGrid: $('resultsGrid'), resultCount: $('resultCount'), clearSelection: $('clearSelection'), compareButton: $('compareButton'), exportButton: $('exportButton'),
    selectedCount: $('selectedCount'), evidenceEmpty: $('evidenceEmpty'), evidenceDetail: $('evidenceDetail'), evidenceImage: $('evidenceImage'), evidenceCrop: $('evidenceCrop'), evidenceBox: $('evidenceBox'),
    evidencePosition: $('evidencePosition'), prevCandidate: $('prevCandidate'), nextCandidate: $('nextCandidate'), evidenceEnlarge: $('evidenceEnlarge'), evidenceMeta: $('evidenceMeta'),
    compareModal: $('compareModal'), compareKicker: $('compareKicker'), compareTitle: $('compareTitle'), compareContent: $('compareContent'), shortcutModal: $('shortcutModal'), toast: $('toast'),
  };

  const fallbackSettings = {
    start: 0, end: 0, coarse_step: 2, dense_step: 0.25, candidate_count: 30,
    identity_threshold: 0.213231, tag_threshold: 0.17, quality_min: 0.15, rescue: true,
    identity_weight: 0.35, condition_weight: 0.3, quality_weight: 0.35,
  };

  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    bindEvents();
    updateWeightTotal();
    renderTags();
    updateSummary();
    updateCaptureAvailability();
    updateCancelControl();
    updateLogoModelStatus();
    updateLogoPreviewControls();
    await loadBootstrap();
    await loadLogoProfiles();
  }

  function bindEvents() {
    els.videoSelect.addEventListener('change', () => inspectSelectedVideo());
    els.videoFile.addEventListener('change', uploadVideo);
    els.referenceFiles.addEventListener('change', uploadReferences);
    els.captureReference.addEventListener('click', captureReferenceFromVideo);
    els.resetCrop.addEventListener('click', resetCrop);
    els.referenceCanvas.addEventListener('pointerdown', beginCrop);
    els.referenceCanvas.addEventListener('pointermove', moveCrop);
    els.referenceCanvas.addEventListener('pointerup', endCrop);
    els.referenceCanvas.addEventListener('pointercancel', endCrop);
    els.wantedTab.addEventListener('click', () => setIntent('wanted'));
    els.avoidedTab.addEventListener('click', () => setIntent('avoided'));
    els.scopeButtons.forEach((button) => button.addEventListener('click', () => setScope(button.dataset.scope)));
    els.tagSearch.addEventListener('input', scheduleVocabularySearch);
    els.tagSearch.addEventListener('keydown', handleTagKeydown);
    els.analyzeButton.addEventListener('click', startAnalysis);
    els.cancelJob.addEventListener('click', cancelAnalysis);
    els.compareButton.addEventListener('click', openCompare);
    els.clearSelection.addEventListener('click', clearSelection);
    els.exportButton.addEventListener('click', exportSelected);
    els.logoRemovalToggle.addEventListener('change', handleLogoRemovalChange);
    els.logoProfileSelect.addEventListener('change', handleLogoProfileChange);
    els.prevCandidate.addEventListener('click', () => stepCandidate(-1));
    els.nextCandidate.addEventListener('click', () => stepCandidate(1));
    els.evidenceEnlarge.addEventListener('click', () => openSingleCandidate(state.activeCandidateIndex));
    els.evidenceImage.addEventListener('click', () => openSingleCandidate(state.activeCandidateIndex));
    els.evidenceCrop.addEventListener('click', () => openSingleCandidate(state.activeCandidateIndex));
    els.shortcutButton.addEventListener('click', () => openModal(els.shortcutModal));
    document.querySelectorAll('[data-close-modal="true"]').forEach((node) => node.addEventListener('click', closeModals));
    [els.identityWeight, els.conditionWeight, els.qualityWeight].forEach((input) => input.addEventListener('input', updateWeightTotal));
    document.addEventListener('click', (event) => {
      if (!event.target.closest('.tag-search-wrap') && !event.target.closest('#tagSuggestions')) hideSuggestions();
    });
    document.addEventListener('keydown', handleGlobalKeydown);
    window.addEventListener('resize', () => { if (state.selectedReferenceId) drawCrop(); });
  }

  async function loadBootstrap() {
    try {
      const data = await api('/api/bootstrap');
      state.bootstrap = data || {};
      state.aliases = data && data.aliases && typeof data.aliases === 'object' ? data.aliases : {};
      applySettings(data && (data.settings || data.defaults && data.defaults.settings));
      populateVideos(Array.isArray(data && data.videos) ? data.videos : []);
      setModelStatus(data && data.models);
      updateSummary();
    } catch (error) {
      setModelStatus({ ready: false, details: 'bootstrap を読み込めませんでした' }, true);
      showNotice(`初期設定を取得できませんでした: ${error.message}`, 'error', 0);
      populateVideos([]);
    }
  }

  async function loadLogoProfiles() {
    const token = ++state.logoProfilesToken;
    try {
      const data = await api('/api/logo-profiles');
      if (token !== state.logoProfilesToken) return;
      state.logoProfiles = normalizeLogoProfiles(data && data.profiles);
      state.logoProfileRoot = typeof (data && data.profile_root) === 'string' ? data.profile_root : '';
      state.logoModel = normalizeLogoModel(data && data.model);
      state.logoDefaultEnabled = typeof (data && data.default_enabled) === 'boolean' ? data.default_enabled : true;
      state.logoDefaultLoaded = true;
      state.logoProfilesLoaded = true;
      if (!state.logoChoiceTouched) {
        els.logoRemovalToggle.checked = state.logoDefaultEnabled;
      }
      populateLogoProfiles();
      updateLogoModelStatus();
      updateLogoPreviewControls();
    } catch (error) {
      if (token !== state.logoProfilesToken) return;
      state.logoProfiles = [];
      state.logoProfileRoot = '';
      state.logoModel = { ready: false, details: error && error.message ? error.message : 'ロゴ情報を取得できませんでした' };
      state.logoDefaultLoaded = true;
      state.logoProfilesLoaded = true;
      populateLogoProfiles();
      updateLogoModelStatus(true);
      updateLogoPreviewControls();
    }
  }

  function normalizeLogoProfiles(rawProfiles) {
    if (!Array.isArray(rawProfiles)) return [];
    const seen = new Set();
    return rawProfiles.map((raw) => {
      if (!raw || typeof raw !== 'object') return null;
      const id = String(raw.id ?? raw.profile_id ?? '').trim();
      if (!id || seen.has(id)) return null;
      seen.add(id);
      return {
        id,
        name: String(raw.name ?? raw.label ?? id),
        referenceSize: raw.reference_size ?? raw.referenceSize ?? '',
        ready: raw.ready !== false,
        details: raw.details ?? '',
      };
    }).filter(Boolean);
  }

  function normalizeLogoModel(rawModel) {
    if (!rawModel || typeof rawModel !== 'object') return { ready: false, details: '' };
    return { ready: rawModel.ready === true, details: rawModel.details ?? rawModel.message ?? '' };
  }

  function populateLogoProfiles() {
    const previous = state.logoSelectedProfileId;
    els.logoProfileSelect.innerHTML = '<option value="">自動（登録済みから検出）</option>';
    state.logoProfiles.forEach((profile) => {
      const option = document.createElement('option');
      option.value = profile.id;
      option.textContent = `${profile.name}（${profile.id}）`;
      option.disabled = profile.ready === false;
      if (profile.ready === false) option.textContent += ' · 未準備';
      els.logoProfileSelect.appendChild(option);
    });
    const selected = state.logoProfiles.some((profile) => profile.id === previous && profile.ready !== false) ? previous : '';
    state.logoSelectedProfileId = selected;
    els.logoProfileSelect.value = selected;
    els.logoProfileEmpty.classList.toggle('hidden', state.logoProfiles.length > 0);
    if (!state.logoProfiles.length) {
      els.logoProfileEmpty.textContent = 'ロゴ未登録: 新しいロゴはSkillで登録';
    }
    invalidateLogoPreview();
  }

  function updateLogoModelStatus(loadFailed = false) {
    if (!state.logoProfilesLoaded) {
      els.logoModelBadge.className = 'ready-badge';
      els.logoModelBadge.textContent = '確認中';
      els.logoProfileStatus.textContent = 'ローカルのロゴ情報を確認しています。';
      return;
    }
    const ready = Boolean(state.logoModel && state.logoModel.ready);
    els.logoModelBadge.className = `ready-badge${ready ? ' ready' : ''}`;
    els.logoModelBadge.textContent = ready ? '準備完了' : '要セットアップ';
    if (loadFailed) {
      els.logoProfileStatus.textContent = 'ロゴ情報を取得できません。ロゴ除去は適用せず、元フレームを保存します。';
    } else if (!ready) {
      const detail = modelDetailText(state.logoModel);
      els.logoProfileStatus.textContent = `ローカルのロゴモデルが未準備です。${detail}`;
    } else if (!state.logoProfiles.length) {
      els.logoProfileStatus.textContent = '登録済みロゴプロファイルがありません。';
    } else {
      els.logoProfileStatus.textContent = 'ローカル登録済みプロファイルから検出します。';
    }
  }

  function handleLogoRemovalChange() {
    state.logoChoiceTouched = true;
    invalidateLogoPreview();
    updateLogoPreviewControls();
  }

  function handleLogoProfileChange() {
    state.logoSelectedProfileId = String(els.logoProfileSelect.value || '');
    invalidateLogoPreview();
    updateLogoPreviewControls();
  }

  function selectedLogoProfiles() {
    const id = String(state.logoSelectedProfileId || els.logoProfileSelect.value || '').trim();
    return id ? [id] : [];
  }

  function logoRemovalEnabled() {
    return Boolean(els.logoRemovalToggle && els.logoRemovalToggle.checked);
  }

  function invalidateLogoPreview() {
    state.logoPreviewToken += 1;
    state.logoPreviewContext = null;
    state.logoPreviewData = null;
    const button = document.getElementById('logoPreviewButton');
    if (button) setBusy(button, false);
    const status = document.getElementById('logoPreviewStatus');
    if (status) status.textContent = '';
    const result = document.getElementById('logoPreviewResult');
    if (result) result.classList.add('hidden');
  }

  function updateLogoPreviewControls() {
    const button = document.getElementById('logoPreviewButton');
    if (!button) return;
    const setting = document.getElementById('logoPreviewSetting');
    if (setting) setting.textContent = logoRemovalEnabled() ? '現在の書き出し設定: ロゴ除去オン' : '現在の書き出し設定: ロゴ除去なし';
    const enabled = logoRemovalEnabled() && Boolean(state.logoModel && state.logoModel.ready) && button.getAttribute('aria-busy') !== 'true';
    button.disabled = !enabled;
    if (!logoRemovalEnabled()) button.title = '「保存時にロゴを除去」をオンにするとプレビューできます';
    else if (!(state.logoModel && state.logoModel.ready)) button.title = 'ローカルのロゴモデルが未準備のためプレビューできません';
    else button.title = '';
  }

  function setModelStatus(models, failed = false) {
    const ready = Boolean(models && models.ready);
    els.modelStatus.className = `status-pill ${failed ? 'status-error' : ready ? 'status-ready' : 'status-off'}`;
    els.modelStatus.innerHTML = `<span class="status-dot"></span><span>${failed ? '接続エラー' : ready ? 'モデル準備完了' : '要セットアップ'}</span>`;
    els.readyBadge.className = `ready-badge${ready ? ' ready' : ''}`;
    els.readyBadge.textContent = ready ? '準備完了' : '要セットアップ';
    if (!ready && !failed) {
      const detail = modelDetailText(models);
      showNotice(`モデルが未準備です。setup.bat を実行してください。${detail} 入力設定は先に進められます。`, 'warning', 0);
    }
  }

  function modelDetailText(models) {
    const details = models && models.details;
    if (!details) return 'ローカルモデルを準備してください。';
    if (typeof details === 'string') return details;
    if (typeof details !== 'object') return String(details);
    for (const key of ['error', 'reason', 'message', 'detail']) {
      const value = details[key];
      if (typeof value === 'string' && value.trim()) return value.trim();
      if (value && typeof value === 'object') {
        const nested = modelDetailText({ details: value });
        if (nested && !nested.includes('ローカルモデルを準備してください')) return nested;
      }
    }
    if (details.model_root) return `モデルルート: ${displayModelValue(details.model_root)}`;
    return 'ローカルモデルを準備してください。';
  }

  function displayModelValue(value) {
    if (typeof value === 'string') return value;
    if (value === null || value === undefined) return '';
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
    try { return JSON.stringify(value); } catch (_) { return '詳細を確認してください。'; }
  }

  function applySettings(settings) {
    const source = Object.assign({}, fallbackSettings, settings || {});
    setInputValue(els.startTime, source.start);
    setInputValue(els.endTime, source.end);
    setInputValue(els.coarseStep, source.coarse_step);
    setInputValue(els.denseStep, source.dense_step);
    setInputValue(els.candidateCount, source.candidate_count);
    setInputValue(els.identityThreshold, source.identity_threshold);
    setInputValue(els.tagThreshold, source.tag_threshold);
    setInputValue(els.qualityMin, source.quality_min);
    setInputValue(els.identityWeight, source.identity_weight);
    setInputValue(els.conditionWeight, source.condition_weight);
    setInputValue(els.qualityWeight, source.quality_weight);
    els.rescue.checked = source.rescue !== false;
    if (Object.prototype.hasOwnProperty.call(source, 'logo_removal') && !state.logoChoiceTouched) {
      els.logoRemovalToggle.checked = source.logo_removal !== false;
    }
    if (Array.isArray(source.logo_profiles) && source.logo_profiles.length && !state.logoChoiceTouched) {
      state.logoSelectedProfileId = String(source.logo_profiles[0]);
    }
    updateWeightTotal();
  }

  function setInputValue(input, value) { if (value !== undefined && value !== null && value !== '') input.value = value; }

  function populateVideos(videos) {
    state.videos = videos;
    els.videoSelect.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = videos.length ? '動画を選択…' : '登録済み動画がありません';
    els.videoSelect.appendChild(placeholder);
    videos.forEach((video) => {
      const path = typeof video === 'string' ? video : video && (video.path || video.id || video.video_id);
      if (!path) return;
      const option = document.createElement('option');
      option.value = path;
      option.textContent = (typeof video === 'object' && (video.name || video.filename)) || basename(path);
      option.dataset.path = path;
      els.videoSelect.appendChild(option);
    });
    if (videos.length === 1) {
      els.videoSelect.value = els.videoSelect.options[1].value;
      inspectSelectedVideo();
    }
  }

  async function inspectSelectedVideo() {
    const path = els.videoSelect.value;
    const token = ++state.videoRequestToken;
    if (!path) {
      state.currentVideo = null;
      state.videoLoading = false;
      updateVideoView();
      return;
    }
    state.currentVideo = null;
    state.videoLoading = true;
    updateVideoView();
    updateCaptureAvailability();
    setBusy(els.videoSelect, true);
    try {
      const data = await api('/api/videos/inspect', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path }) });
      if (token !== state.videoRequestToken) return;
      setCurrentVideo(data, null, path);
      clearGlobalNotice();
      showToast('動画を読み込みました');
    } catch (error) {
      if (token === state.videoRequestToken) showNotice(`動画情報を取得できませんでした: ${error.message}`, 'error');
    } finally {
      if (token === state.videoRequestToken) {
        state.videoLoading = false;
        updateCaptureAvailability();
        setBusy(els.videoSelect, false);
      }
    }
  }

  async function uploadVideo(event) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    ++state.videoRequestToken;
    state.videoLoading = true;
    updateCaptureAvailability();
    setBusy(els.videoFile, true);
    try {
      const data = await api('/api/videos/upload', { method: 'POST', body: form });
      const objectUrl = URL.createObjectURL(file);
      setCurrentVideo(data, objectUrl, data && data.info && data.info.path ? data.info.path : file.name);
      const option = document.createElement('option');
      option.value = (data && (data.video_id || data.id)) || file.name;
      option.textContent = `アップロード: ${file.name}`;
      option.dataset.path = (data && data.info && data.info.path) || file.name;
      els.videoSelect.appendChild(option);
      els.videoSelect.value = option.value;
      clearGlobalNotice();
      showToast('動画を追加しました');
    } catch (error) {
      showNotice(`動画を追加できませんでした: ${error.message}`, 'error');
    } finally {
      state.videoLoading = false;
      updateCaptureAvailability();
      setBusy(els.videoFile, false);
      event.target.value = '';
    }
  }

  function setCurrentVideo(data, objectUrl, fallbackPath) {
    const info = data && data.info ? data.info : data || {};
    const videoId = data && (data.video_id || data.id) || info.video_id || fallbackPath;
    state.currentVideo = { video_id: videoId, info, path: info.path || fallbackPath, name: info.name || basename(info.path || fallbackPath || '') };
    if (state.videoObjectUrl) URL.revokeObjectURL(state.videoObjectUrl);
    state.videoObjectUrl = objectUrl;
    state.videoLoading = false;
    updateVideoView();
    updateCaptureAvailability();
  }

  function updateVideoView() {
    const video = state.currentVideo;
    if (!video) {
      els.videoMeta.className = 'meta-card muted';
      els.videoMeta.textContent = '動画を選択すると情報が表示されます';
      els.videoPreview.classList.add('hidden');
      els.summaryVideo.textContent = '未選択';
      updateCaptureAvailability();
      return;
    }
    const info = video.info || {};
    els.videoMeta.className = 'meta-card';
    els.videoMeta.innerHTML = `<div class="meta-line"><strong>${escapeHTML(video.name || basename(video.path || '動画'))}</strong></div><div class="meta-line">${escapeHTML(formatVideoInfo(info))}</div>`;
    els.summaryVideo.textContent = video.name || basename(video.path || '動画');
    if (state.videoObjectUrl) {
      els.videoPreview.src = state.videoObjectUrl;
      els.videoPreview.classList.remove('hidden');
    } else {
      els.videoPreview.removeAttribute('src');
      els.videoPreview.classList.add('hidden');
    }
  }

  async function uploadReferences(event) {
    const files = [...(event.target.files || [])];
    if (!files.length) return;
    const form = new FormData();
    files.forEach((file) => form.append('files', file));
    setBusy(els.referenceFiles, true);
    try {
      const data = await api('/api/references', { method: 'POST', body: form });
      const refs = Array.isArray(data && data.references) ? data.references : [];
      refs.forEach((reference) => addReference(reference));
      if (refs.length) showToast(`${refs.length}枚の参照画像を追加しました`);
      else showNotice('参照画像が見つかりませんでした。', 'warning');
    } catch (error) {
      showNotice(`参照画像を追加できませんでした: ${error.message}`, 'error');
    } finally {
      setBusy(els.referenceFiles, false);
      event.target.value = '';
    }
  }

  function addReference(raw) {
    if (!raw) return;
    const id = String(raw.id || raw.reference_id || `ref-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    const reference = {
      id, url: raw.url || raw.preview_url || raw.path || '', width: raw.width || 0, height: raw.height || 0,
      name: raw.name || raw.filename || `参照 ${state.references.length + 1}`, box: null, image: null,
    };
    state.references.push(reference);
    if (!state.selectedReferenceId) state.selectedReferenceId = id;
    loadReferenceImage(reference).then(() => { renderReferences(); drawCrop(); });
    renderReferences();
    updateSummary();
    drawCrop();
  }

  function loadReferenceImage(reference) {
    if (reference.image || !reference.url) return Promise.resolve(reference.image);
    return new Promise((resolve) => {
      const image = new Image();
      image.onload = () => { reference.image = image; if (!reference.width) reference.width = image.naturalWidth; if (!reference.height) reference.height = image.naturalHeight; resolve(image); };
      image.onerror = () => resolve(null);
      image.src = reference.url;
    });
  }

  function renderReferences() {
    els.referenceCount.textContent = String(state.references.length);
    if (!state.references.length) {
      els.referenceList.className = 'reference-list empty-state';
      els.referenceList.textContent = '参照画像がありません';
      els.cropEmpty.classList.remove('hidden');
      els.cropStage.classList.add('hidden');
      els.selectedReferenceName.classList.add('hidden');
      return;
    }
    els.referenceList.className = 'reference-list';
    els.referenceList.innerHTML = '';
    state.references.forEach((reference) => {
      const item = document.createElement('button');
      item.type = 'button';
      item.className = `reference-item${reference.id === state.selectedReferenceId ? ' active' : ''}`;
      item.title = '対象範囲を編集';
      item.innerHTML = `<img class="reference-thumb" src="${escapeAttribute(reference.url)}" alt=""><span class="reference-name">${escapeHTML(reference.name)}<small>${reference.width && reference.height ? `${reference.width} × ${reference.height}` : '画像'}</small></span><span class="reference-check">${reference.id === state.selectedReferenceId ? '●' : '○'}</span>`;
      item.addEventListener('click', () => selectReference(reference.id));
      els.referenceList.appendChild(item);
    });
  }

  async function selectReference(id) {
    const reference = state.references.find((item) => item.id === String(id));
    if (!reference) return;
    state.selectedReferenceId = reference.id;
    renderReferences();
    await loadReferenceImage(reference);
    drawCrop();
  }

  async function captureReferenceFromVideo() {
    if (!state.currentVideo || !state.currentVideo.video_id) {
      showNotice('先に動画を選択してください。', 'warning');
      return;
    }
    const time = Math.max(0, Number(els.referenceTime.value) || 0);
    state.captureLoading = true;
    setBusy(els.captureReference, true);
    updateCaptureAvailability();
    try {
      const data = await api(`/api/videos/${encodeURIComponent(state.currentVideo.video_id)}/reference`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ time }) });
      addReference(data && (data.reference || data));
      clearGlobalNotice();
      showToast(`${formatTime(time)} のフレームを参照に追加しました`);
    } catch (error) {
      showNotice(`フレームを取得できませんでした: ${error.message}`, 'error');
    } finally {
      state.captureLoading = false;
      setBusy(els.captureReference, false);
      updateCaptureAvailability();
    }
  }

  function drawCrop() {
    const reference = state.references.find((item) => item.id === state.selectedReferenceId);
    if (!reference || !reference.image) {
      if (reference && reference.url) loadReferenceImage(reference).then(drawCrop);
      els.cropEmpty.classList.toggle('hidden', Boolean(reference));
      els.cropStage.classList.toggle('hidden', !reference);
      if (reference) els.selectedReferenceName.classList.remove('hidden');
      if (reference) els.selectedReferenceName.textContent = `${reference.name} · 読み込み中…`;
      return;
    }
    els.cropEmpty.classList.add('hidden');
    els.cropStage.classList.remove('hidden');
    els.selectedReferenceName.classList.remove('hidden');
    els.selectedReferenceName.textContent = `${reference.name} · ドラッグで対象範囲を調整`;
    const stage = els.cropStage.getBoundingClientRect();
    const dpr = Math.max(1, Math.min(2, Number(window.devicePixelRatio) || 1));
    const width = Math.max(1, Math.floor(stage.width));
    const height = Math.max(1, Math.floor(stage.height));
    const backingWidth = Math.max(1, Math.round(width * dpr));
    const backingHeight = Math.max(1, Math.round(height * dpr));
    els.referenceCanvas.width = backingWidth;
    els.referenceCanvas.height = backingHeight;
    els.referenceCanvas.style.width = `${width}px`;
    els.referenceCanvas.style.height = `${height}px`;
    const ctx = els.referenceCanvas.getContext('2d');
    // Draw in backing-store pixels explicitly. This avoids browser-specific
    // double scaling when the canvas is CSS-sized on a high-DPI display.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#111119';
    ctx.fillRect(0, 0, backingWidth, backingHeight);
    const cssImageRect = getImageRect(reference, width, height);
    const imageRect = { x: cssImageRect.x * dpr, y: cssImageRect.y * dpr, width: cssImageRect.width * dpr, height: cssImageRect.height * dpr };
    ctx.drawImage(reference.image, imageRect.x, imageRect.y, imageRect.width, imageRect.height);
    const box = reference.box || [0, 0, 1, 1];
    const x = imageRect.x + box[0] * imageRect.width;
    const y = imageRect.y + box[1] * imageRect.height;
    const w = (box[2] - box[0]) * imageRect.width;
    const h = (box[3] - box[1]) * imageRect.height;
    ctx.fillStyle = 'rgba(5,5,10,.42)';
    ctx.fillRect(imageRect.x, imageRect.y, imageRect.width, imageRect.height);
    // Restore the selected area from the full fitted image. The imageRect is
    // a destination rectangle; using it as the source rectangle would zoom
    // into the upper-left of the reference on every redraw.
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, w, h);
    ctx.clip();
    ctx.drawImage(reference.image, imageRect.x, imageRect.y, imageRect.width, imageRect.height);
    ctx.restore();
    ctx.fillStyle = reference.box ? 'rgba(167,123,255,.14)' : 'rgba(112,217,194,.08)';
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = reference.box ? '#c4a6ff' : '#70d9c2';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x + .5, y + .5, Math.max(0, w - 1), Math.max(0, h - 1));
    updateCropReadout(reference.box);
  }

  function getImageRect(reference, width, height) {
    const imageWidth = reference.image.naturalWidth || reference.width || width;
    const imageHeight = reference.image.naturalHeight || reference.height || height;
    const ratio = Math.min(width / imageWidth, height / imageHeight);
    const drawnWidth = imageWidth * ratio;
    const drawnHeight = imageHeight * ratio;
    return { x: (width - drawnWidth) / 2, y: (height - drawnHeight) / 2, width: drawnWidth, height: drawnHeight };
  }

  function beginCrop(event) {
    const reference = state.references.find((item) => item.id === state.selectedReferenceId);
    if (!reference || !reference.image) return;
    const point = canvasPoint(event, reference);
    if (!point) return;
    state.cropPointer = { start: point, moved: false };
    els.referenceCanvas.setPointerCapture?.(event.pointerId);
  }

  function moveCrop(event) {
    if (!state.cropPointer) return;
    const reference = state.references.find((item) => item.id === state.selectedReferenceId);
    const point = canvasPoint(event, reference);
    if (!point) return;
    const start = state.cropPointer.start;
    if (Math.abs(point.x - start.x) + Math.abs(point.y - start.y) > 0.01) state.cropPointer.moved = true;
    if (!state.cropPointer.moved) return;
    reference.box = normalizeBox([Math.min(start.x, point.x), Math.min(start.y, point.y), Math.max(start.x, point.x), Math.max(start.y, point.y)]);
    drawCrop();
  }

  function endCrop() {
    if (state.cropPointer && !state.cropPointer.moved) resetCrop();
    state.cropPointer = null;
  }

  function canvasPoint(event, reference) {
    if (!reference || !reference.image) return null;
    const rect = els.referenceCanvas.getBoundingClientRect();
    const width = rect.width; const height = rect.height;
    const point = { x: (event.clientX - rect.left) / width, y: (event.clientY - rect.top) / height };
    const imageRect = getImageRect(reference, width, height);
    return { x: clamp((point.x * width - imageRect.x) / imageRect.width, 0, 1), y: clamp((point.y * height - imageRect.y) / imageRect.height, 0, 1) };
  }

  function resetCrop() {
    const reference = state.references.find((item) => item.id === state.selectedReferenceId);
    if (reference) { reference.box = null; drawCrop(); }
  }

  function updateCropReadout(box) {
    if (!box) { els.cropReadout.textContent = '全体を使用'; return; }
    const width = Math.round((box[2] - box[0]) * 100); const height = Math.round((box[3] - box[1]) * 100);
    els.cropReadout.textContent = `${Math.round(box[0] * 100)}%, ${Math.round(box[1] * 100)}% · ${width} × ${height}%`;
  }

  function setIntent(intent) {
    state.intent = intent;
    els.wantedTab.classList.toggle('active', intent === 'wanted');
    els.avoidedTab.classList.toggle('active', intent === 'avoided');
    els.wantedTab.setAttribute('aria-selected', intent === 'wanted' ? 'true' : 'false');
    els.avoidedTab.setAttribute('aria-selected', intent === 'avoided' ? 'true' : 'false');
    els.bucketTitle.textContent = intent === 'wanted' ? '欲しいタグ' : '避けたいタグ';
  }

  function setScope(scope) {
    if (scope !== 'subject' && scope !== 'frame') return;
    state.scope = scope;
    els.scopeButtons.forEach((button) => button.classList.toggle('active', button.dataset.scope === scope));
  }

  function scheduleVocabularySearch() {
    clearTimeout(state.searchTimer);
    setTagWarning('');
    const query = els.tagSearch.value.trim();
    if (!query) { hideSuggestions(); return; }
    state.searchTimer = setTimeout(() => searchVocabulary(query), 210);
  }

  async function searchVocabulary(query) {
    const token = ++state.searchToken;
    els.tagSearchSpinner.classList.remove('hidden');
    try {
      const data = await api(`/api/vocabulary?q=${encodeURIComponent(query)}`);
      if (token !== state.searchToken) return;
      if (data && data.aliases && typeof data.aliases === 'object') Object.assign(state.aliases, data.aliases);
      let tags = Array.isArray(data && data.tags) ? data.tags : [];
      // Some manifests spell a canonical tag with spaces while a legacy alias
      // uses underscores (or hyphens). If the server's alias filter cannot
      // match that spelling, make one unfiltered dictionary read and match by
      // the normalized form on the client.
      const aliasTarget = resolveAlias(query);
      if (!tags.length) {
        const complete = await api('/api/vocabulary');
        if (token !== state.searchToken) return;
        const completeTags = Array.isArray(complete && complete.tags) ? complete.tags : [];
        if (complete && complete.aliases && typeof complete.aliases === 'object') Object.assign(state.aliases, complete.aliases);
        rememberVocabularyTags(completeTags);
        const queryKey = normalizeTagName(query);
        const aliasKey = normalizeTagName(aliasTarget);
        tags = completeTags.filter((tag) => {
          if (!tag) return false;
          const nameKey = normalizeTagName(tag.name || tag.tag);
          return (aliasKey && nameKey === aliasKey) || Boolean(queryKey && nameKey.includes(queryKey));
        });
      }
      rememberVocabularyTags(tags);
      renderSuggestions(tags, query);
    } catch (error) {
      if (token === state.searchToken) showNotice(`タグ辞書を取得できませんでした: ${error.message}`, 'error');
    } finally {
      if (token === state.searchToken) els.tagSearchSpinner.classList.add('hidden');
    }
  }

  function renderSuggestions(tags, query) {
    els.tagSuggestions.innerHTML = '';
    const matches = tags.filter(isConditionTag).slice(0, 12);
    const aliasMatch = resolveAlias(query);
    if (!matches.length && aliasMatch) {
      matches.push({ name: aliasMatch, alias: query, category: 'alias' });
      state.vocabulary.set(aliasMatch, matches[matches.length - 1]);
    }
    if (!matches.length) {
      const blocked = tags.some((tag) => tag && !isConditionTag(tag));
      els.tagSuggestions.innerHTML = `<div class="suggestion"><span class="suggestion-main"><span class="suggestion-name">${blocked ? '一般タグの条件として使えないカテゴリです' : '一致する既知タグがありません'}</span></span><span class="suggestion-scope">Enterで警告</span></div>`;
    } else {
      matches.forEach((tag) => {
        const name = String(tag.name || tag.tag || '');
        const alias = tag.alias || reverseAlias(name, query);
        const item = document.createElement('button'); item.type = 'button'; item.className = 'suggestion'; item.setAttribute('role', 'option'); item.dataset.tag = name;
        const scopeLabel = formatScope(isConditionTag(tag) ? state.scope : tag.scope);
        item.innerHTML = `<span class="suggestion-main"><span class="suggestion-name">${escapeHTML(name)}${alias ? `<span class="suggestion-alias">（${escapeHTML(alias)}）</span>` : ''}</span></span><span class="suggestion-scope">${escapeHTML(scopeLabel)}</span>`;
        item.addEventListener('click', () => addTag(name, state.intent, state.scope, alias));
        els.tagSuggestions.appendChild(item);
      });
    }
    els.tagSuggestions.classList.remove('hidden');
  }

  function handleTagKeydown(event) {
    if (event.key === 'Escape') { hideSuggestions(); return; }
    if (event.key !== 'Enter') return;
    event.preventDefault();
    const first = els.tagSuggestions.querySelector('.suggestion[data-tag]');
    if (first) { addTag(first.dataset.tag, state.intent, state.scope, reverseAlias(first.dataset.tag, els.tagSearch.value.trim())); return; }
    const query = els.tagSearch.value.trim();
    if (query) setTagWarning(`「${query}」は既知のタグではありません。候補から選択してください。`);
  }

  function addTag(name, intent, scope, alias) {
    const canonical = resolveKnownTag(name) || String(name || '').trim();
    if (!canonical || !isKnownTag(canonical)) {
      setTagWarning(`「${name}」は既知のタグではありません。`);
      return;
    }
    const bucket = intent === 'avoided' ? state.avoided : state.wanted;
    const tagInfo = state.vocabulary.get(canonical);
    // General condition tags may be evaluated on either the subject crop or
    // the full frame; preserve the user's explicit scope. A model category
    // hint is still shown by the suggestion list, but it must not silently
    // replace the requested crop/frame target.
    const knownScope = isConditionTag(tagInfo) ? scope : (tagInfo && (tagInfo.scope === 'subject' || tagInfo.scope === 'frame') ? tagInfo.scope : scope);
    if (knownScope !== scope) setTagWarning(`「${canonical}」は${knownScope === 'subject' ? '対象' : 'フレーム'}タグのため、その範囲で追加しました。`);
    if (!bucket.some((item) => item.tag === canonical && item.scope === knownScope)) bucket.push({ tag: canonical, scope: knownScope, alias: alias || reverseAlias(canonical, '') });
    els.tagSearch.value = '';
    hideSuggestions();
    if (knownScope === scope) setTagWarning('');
    renderTags();
    updateSummary();
  }

  function removeTag(intent, index) {
    const bucket = intent === 'avoided' ? state.avoided : state.wanted;
    bucket.splice(index, 1); renderTags(); updateSummary();
  }

  function renderTags() {
    renderTagBucket(els.wantedTags, state.wanted, 'wanted');
    renderTagBucket(els.avoidedTags, state.avoided, 'avoided');
    els.wantedCount.textContent = String(state.wanted.length); els.avoidedCount.textContent = String(state.avoided.length);
  }

  function renderTagBucket(container, tags, intent) {
    container.innerHTML = '';
    if (!tags.length) { container.innerHTML = `<span class="chips-empty">${intent === 'wanted' ? '検索結果から追加できます' : '必要なときだけ追加'}</span>`; return; }
    tags.forEach((item, index) => {
      const chip = document.createElement('span'); chip.className = `tag-chip${intent === 'avoided' ? ' avoided' : ''}`;
      chip.innerHTML = `<span>${escapeHTML(item.alias ? `${item.tag} · ${item.alias}` : item.tag)}</span><em>${item.scope === 'subject' ? '対象' : 'フレーム'}</em><button type="button" aria-label="${escapeAttribute(item.tag)}を削除">×</button>`;
      chip.querySelector('button').addEventListener('click', () => removeTag(intent, index)); container.appendChild(chip);
    });
  }

  function setTagWarning(message) {
    if (!message) { els.tagWarning.classList.add('hidden'); els.tagWarning.textContent = ''; return; }
    els.tagWarning.classList.remove('hidden'); els.tagWarning.textContent = message;
  }

  function hideSuggestions() { els.tagSuggestions.classList.add('hidden'); }

  async function startAnalysis() {
    const error = validateBeforeAnalysis();
    if (error) { showNotice(error, 'warning'); return; }
    const body = {
      video_id: state.currentVideo.video_id,
      references: state.references.map((reference) => ({ id: reference.id, box: reference.box || null })),
      wanted: state.wanted.map((item) => ({ tag: item.tag, scope: item.scope })),
      avoided: state.avoided.map((item) => ({ tag: item.tag, scope: item.scope })),
      settings: collectSettings(),
    };
    setBusy(els.analyzeButton, true); showProgress(0, '解析ジョブを準備中…'); clearResults();
    try {
      const data = await api('/api/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const jobId = data && (data.job_id || data.id);
      if (!jobId) throw new Error('job_id が返されませんでした');
      state.job = { id: String(jobId), status: 'queued' };
      updateCancelControl();
      showProgress(0, 'キューに追加しました…'); pollJob();
    } catch (error) {
      hideProgress(); setBusy(els.analyzeButton, false); updateCancelControl(); showNotice(`解析を開始できませんでした: ${error.message}`, 'error');
    }
  }

  function validateBeforeAnalysis() {
    if (!state.currentVideo || !state.currentVideo.video_id) return '動画を選択してください。';
    if (!state.references.length) return '参照画像を1枚以上追加してください。';
    const count = clamp(Math.round(getNumber(els.candidateCount, 30)), 20, 50);
    els.candidateCount.value = count;
    return '';
  }

  function collectSettings() {
    return {
      start: getNumber(els.startTime, 0), end: getNumber(els.endTime, 0), coarse_step: getNumber(els.coarseStep, 2), dense_step: getNumber(els.denseStep, .25),
      candidate_count: clamp(Math.round(getNumber(els.candidateCount, 30)), 20, 50), identity_threshold: getNumber(els.identityThreshold, .213231), tag_threshold: getNumber(els.tagThreshold, .17), quality_min: getNumber(els.qualityMin, .15), rescue: els.rescue.checked,
      identity_weight: getNumber(els.identityWeight, .35), condition_weight: getNumber(els.conditionWeight, .3), quality_weight: getNumber(els.qualityWeight, .35),
      logo_removal: logoRemovalEnabled(), logo_profiles: selectedLogoProfiles(),
    };
  }

  function pollJob() {
    if (!state.job || state.pollTimer || state.pollInFlight) return;
    const tick = async () => {
      if (!state.job || state.pollInFlight) return;
      const polledJobId = state.job.id;
      state.pollInFlight = true;
      try {
        const data = await api(`/api/jobs/${encodeURIComponent(polledJobId)}`);
        // A cancellation or a newer job may have completed while this GET
        // was in flight. Never let that stale response resurrect polling.
        if (!state.job || state.job.id !== polledJobId || ['cancelled', 'done', 'error'].includes(state.job.status)) return;
        state.job = Object.assign({}, state.job, data);
        const progress = clamp(Number(data.progress) || 0, 0, 1);
        const terminal = ['done', 'cancelled', 'error'].includes(data.status);
        showProgress(progress, state.job.cancelRequested && !terminal ? 'キャンセル要求済み · 処理終了を待機中…' : data.message || statusLabel(data.status));
        if (data.status === 'done') {
          state.candidates = Array.isArray(data.candidates) ? data.candidates : [];
          state.selectedIds.clear(); state.activeCandidateIndex = state.candidates.length ? 0 : -1;
          renderResults(data.warnings || [], data.stats || {}); hideProgress(); setBusy(els.analyzeButton, false); updateCancelControl(); state.pollTimer = null; return;
        }
        if (data.status === 'cancelled') { hideProgress(); setBusy(els.analyzeButton, false); updateCancelControl(); state.pollTimer = null; showToast('解析をキャンセルしました'); return; }
        if (data.status === 'error') { hideProgress(); setBusy(els.analyzeButton, false); updateCancelControl(); state.pollTimer = null; showNotice(data.error || '解析中にエラーが発生しました。', 'error'); return; }
      } catch (error) {
        if (state.job && state.job.cancelRequested) {
          showProgress(clamp(Number(state.job.progress) || 0, 0, 1), 'キャンセル状態を確認中…');
        } else {
          hideProgress(); setBusy(els.analyzeButton, false); updateCancelControl(); state.pollTimer = null; showNotice(`解析状況を取得できませんでした: ${error.message}`, 'error'); return;
        }
      } finally {
        state.pollInFlight = false;
      }
      state.pollTimer = setTimeout(() => { state.pollTimer = null; tick(); }, 850);
    };
    tick();
  }

  async function cancelAnalysis() {
    if (!state.job || !state.job.id || !['queued', 'running'].includes(state.job.status) || state.job.cancelRequested) return;
    const jobId = state.job.id;
    state.job.cancelRequested = true;
    updateCancelControl();
    showProgress(clamp(Number(state.job.progress) || 0, 0, 1), 'キャンセル要求を送信中…');
    try {
      const cancelled = await api(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
      if (!state.job || state.job.id !== jobId) return;
      if (cancelled && cancelled.status === 'done') {
        state.job = Object.assign({}, state.job, cancelled);
        state.candidates = Array.isArray(cancelled.candidates) ? cancelled.candidates : [];
        state.activeCandidateIndex = state.candidates.length ? 0 : -1;
        renderResults(cancelled.warnings || [], cancelled.stats || {});
        hideProgress(); setBusy(els.analyzeButton, false); updateCancelControl(); return;
      }
      if (cancelled && typeof cancelled === 'object') state.job = Object.assign({}, state.job, cancelled, { cancelRequested: true });
      const terminal = state.job && ['done', 'cancelled', 'error'].includes(state.job.status);
      if (terminal) {
        if (state.job.status === 'cancelled') showToast('解析をキャンセルしました');
        if (state.job.status === 'error') showNotice(state.job.error || '解析中にエラーが発生しました。', 'error');
        if (state.job.status === 'done') {
          state.candidates = Array.isArray(state.job.candidates) ? state.job.candidates : [];
          state.activeCandidateIndex = state.candidates.length ? 0 : -1;
          renderResults(state.job.warnings || [], state.job.stats || {});
        }
        hideProgress(); setBusy(els.analyzeButton, false); updateCancelControl();
      } else {
        showProgress(clamp(Number(state.job.progress) || 0, 0, 1), 'キャンセル要求済み · 処理終了を待機中…');
        // If the regular poll was between ticks, ensure it resumes. An
        // in-flight request will schedule its own next tick in its finally
        // path, so this does not create a second polling loop.
        pollJob();
      }
    } catch (error) {
      if (state.job) state.job.cancelRequested = false;
      updateCancelControl();
      showNotice(`キャンセルできませんでした: ${error.message}`, 'error');
    }
  }

  function showProgress(progress, message) {
    els.progressArea.classList.remove('hidden'); els.progressMessage.textContent = message; els.progressPercent.textContent = `${Math.round(progress * 100)}%`; els.progressBar.style.width = `${Math.round(progress * 100)}%`;
  }
  function hideProgress() { els.progressArea.classList.add('hidden'); }
  function clearResults() { invalidateLogoPreview(); state.candidates = []; state.selectedIds.clear(); state.activeCandidateIndex = -1; renderResults(); els.shortageNotice.classList.add('hidden'); }

  function renderResults(warnings = [], stats = {}) {
    const candidates = state.candidates;
    els.resultCount.textContent = `${candidates.length}件`;
    els.resultsEmpty.classList.toggle('hidden', candidates.length > 0);
    els.resultsGrid.classList.toggle('hidden', candidates.length === 0);
    els.resultsGrid.innerHTML = '';
    if (!candidates.length) {
      els.evidenceEmpty.classList.remove('hidden'); els.evidenceDetail.classList.add('hidden');
      els.evidenceEnlarge.disabled = true;
      state.activeCandidateIndex = -1;
    } else {
      candidates.forEach((candidate, index) => els.resultsGrid.appendChild(createCandidateCard(candidate, index)));
      showEvidence(state.activeCandidateIndex >= 0 ? state.activeCandidateIndex : 0);
    }
    const warningText = warnings.length ? warnings.map(humanizeWarning).join(' · ') : shortageMessage(stats, candidates.length);
    els.shortageNotice.textContent = warningText || '';
    els.shortageNotice.classList.toggle('hidden', !warningText);
    updateSelectionControls();
  }

  function createCandidateCard(candidate, index) {
    const id = String(candidate.id ?? index);
    const card = document.createElement('article');
    card.className = `candidate-card${index === state.activeCandidateIndex ? ' active' : ''}${state.selectedIds.has(id) ? ' selected' : ''}`;
    card.dataset.index = index; card.dataset.id = id;
    const preview = candidate.preview_url || candidate.crop_url || '';
    const bbox = normalizeBox(candidate.bbox);
    card.innerHTML = `<div class="candidate-thumb"><img src="${escapeAttribute(preview)}" alt="${escapeAttribute(`候補 ${formatTime(candidate.time)}`)}">${bbox ? `<span class="mini-bbox" style="left:${bbox[0] * 100}%;top:${bbox[1] * 100}%;width:${(bbox[2] - bbox[0]) * 100}%;height:${(bbox[3] - bbox[1]) * 100}%"></span>` : ''}<span class="candidate-score">${formatPercent(candidate.score)}</span><input class="candidate-check" type="checkbox" aria-label="${escapeAttribute(`候補 ${formatTime(candidate.time)}を選択`)}" ${state.selectedIds.has(id) ? 'checked' : ''}></div><div class="candidate-info"><div class="candidate-time">${escapeHTML(formatTime(candidate.time))}</div><div class="candidate-evidence">${escapeHTML(evidenceLine(candidate))}</div><div class="candidate-tags">${tagSnippet(candidate.tags)}</div></div>`;
    card.addEventListener('click', (event) => { if (event.target.closest('.candidate-check')) return; showEvidence(index); });
    card.addEventListener('dblclick', (event) => { if (event.target.closest('.candidate-check')) return; showEvidence(index); openSingleCandidate(index); });
    const checkbox = card.querySelector('.candidate-check'); checkbox.addEventListener('change', () => { toggleCandidate(id, checkbox.checked); });
    return card;
  }

  function toggleCandidate(id, checked) {
    if (checked) state.selectedIds.add(String(id)); else state.selectedIds.delete(String(id));
    updateSelectionControls();
    document.querySelectorAll('.candidate-card').forEach((card) => { if (card.dataset.id === String(id)) card.classList.toggle('selected', checked); });
  }

  function updateSelectionControls() {
    const count = state.selectedIds.size; els.selectedCount.textContent = String(count); els.clearSelection.disabled = count === 0; els.compareButton.disabled = count !== 2; els.exportButton.disabled = count === 0 || !state.job || state.job.status !== 'done';
  }

  function clearSelection() {
    state.selectedIds.clear();
    document.querySelectorAll('.candidate-card').forEach((card) => { card.classList.remove('selected'); const checkbox = card.querySelector('.candidate-check'); if (checkbox) checkbox.checked = false; });
    updateSelectionControls();
  }

  function showEvidence(index) {
    if (!state.candidates.length) return;
    state.activeCandidateIndex = (index + state.candidates.length) % state.candidates.length;
    const candidate = state.candidates[state.activeCandidateIndex];
    els.evidenceEnlarge.disabled = false;
    els.evidenceEmpty.classList.add('hidden'); els.evidenceDetail.classList.remove('hidden');
    els.evidenceImage.src = candidate.preview_url || candidate.crop_url || '';
    els.evidenceCrop.src = candidate.crop_url || candidate.preview_url || '';
    const bbox = normalizeBox(candidate.bbox);
    if (bbox) { els.evidenceBox.classList.remove('hidden'); els.evidenceBox.style.left = `${bbox[0] * 100}%`; els.evidenceBox.style.top = `${bbox[1] * 100}%`; els.evidenceBox.style.width = `${(bbox[2] - bbox[0]) * 100}%`; els.evidenceBox.style.height = `${(bbox[3] - bbox[1]) * 100}%`; } else els.evidenceBox.classList.add('hidden');
    els.evidencePosition.textContent = `${state.activeCandidateIndex + 1} / ${state.candidates.length}`;
    els.evidenceMeta.innerHTML = `<div class="evidence-stat"><span>時刻 / PTS</span><strong>${escapeHTML(formatTime(candidate.time))} · ${escapeHTML(String(candidate.pts ?? '—'))}</strong></div><div class="evidence-stat"><span>候補 ID / スコア</span><strong>${escapeHTML(String(candidate.id ?? state.activeCandidateIndex))} · ${formatPercent(candidate.score)}</strong></div><div class="evidence-stat"><span>同一性 / 距離</span><strong>${formatPercent(candidate.identity)} · ${formatNumber(candidate.distance)}</strong></div><div class="evidence-stat"><span>画質 / 状態</span><strong>${formatPercent(candidate.quality)} · ${formatPercent(candidate.condition)}</strong></div>${tagEvidence(candidate.tags)}`;
    document.querySelectorAll('.candidate-card').forEach((card) => card.classList.toggle('active', Number(card.dataset.index) === state.activeCandidateIndex));
  }

  function stepCandidate(delta) {
    if (!state.candidates.length) return;
    showEvidence(state.activeCandidateIndex + delta);
    if (!els.compareModal.classList.contains('hidden') && els.compareContent.classList.contains('single-preview')) renderSingleCandidate(state.candidates[state.activeCandidateIndex]);
  }

  function openSingleCandidate(index) {
    if (!state.candidates.length) return;
    showEvidence(index >= 0 ? index : 0);
    els.compareKicker.textContent = 'PREVIEW';
    els.compareTitle.textContent = '候補を拡大';
    els.compareContent.className = 'compare-content single-preview';
    renderSingleCandidate(state.candidates[state.activeCandidateIndex]);
    openModal(els.compareModal);
  }

  function renderSingleCandidate(candidate) {
    if (!candidate) return;
    invalidateLogoPreview();
    const preview = candidate.preview_url || candidate.crop_url || '';
    const crop = candidate.crop_url || preview;
    const bbox = normalizeBox(candidate.bbox);
    const candidateId = String(candidate.id ?? state.activeCandidateIndex);
    els.compareContent.innerHTML = `<div class="single-preview-pane" data-candidate-id="${escapeAttribute(candidateId)}"><div class="single-preview-images"><div class="single-preview-main"><img src="${escapeAttribute(preview)}" alt="候補 ${escapeAttribute(formatTime(candidate.time))}の元フレーム"><span class="single-preview-bbox" style="${bbox ? `left:${bbox[0] * 100}%;top:${bbox[1] * 100}%;width:${(bbox[2] - bbox[0]) * 100}%;height:${(bbox[3] - bbox[1]) * 100}%` : 'display:none'}"></span></div><div class="single-preview-target"><span>対象 crop（解析領域・元映像）</span><img src="${escapeAttribute(crop)}" alt="候補 ${escapeAttribute(formatTime(candidate.time))}の対象crop（元映像）"></div></div><div class="single-preview-actions"><button id="logoPreviewButton" class="ghost-button" type="button">ロゴ除去をプレビュー</button><span id="logoPreviewSetting" class="logo-preview-setting"></span></div><div id="logoPreviewStatus" class="logo-preview-status" role="status"></div><div id="logoPreviewResult" class="logo-preview-result hidden"><div class="logo-preview-views"><figure><figcaption>元フレーム（保存用）</figcaption><img id="logoPreviewOriginal" alt="元フレーム（保存用）"></figure><figure><figcaption id="logoPreviewCleanLabel">AI補完後（ロゴ除去）</figcaption><img id="logoPreviewClean" alt="AI補完後（ロゴ除去）"><div id="logoPreviewCleanPlaceholder" class="logo-preview-placeholder hidden">元フレームを保持しました</div></figure></div><div id="logoPreviewMeta" class="logo-preview-meta"></div><div id="logoPreviewWarnings" class="logo-preview-warnings"></div></div><div class="single-preview-meta"><strong>${escapeHTML(formatTime(candidate.time))}</strong><span>PTS ${escapeHTML(String(candidate.pts ?? '—'))}</span><span>ID ${escapeHTML(String(candidate.id ?? '—'))}</span><span>スコア ${formatPercent(candidate.score)}</span><p>${escapeHTML(evidenceLine(candidate))}</p>${tagEvidence(candidate.tags)}</div></div>`;
    const button = document.getElementById('logoPreviewButton');
    button.addEventListener('click', () => previewLogoForCandidate(candidate));
    updateLogoPreviewControls();
  }

  async function previewLogoForCandidate(candidate) {
    if (!candidate || !state.job || !state.job.id) return;
    const button = document.getElementById('logoPreviewButton');
    const status = document.getElementById('logoPreviewStatus');
    if (!logoRemovalEnabled()) {
      if (status) status.textContent = '「保存時にロゴを除去」をオンにするとプレビューできます。';
      updateLogoPreviewControls();
      return;
    }
    if (!(state.logoModel && state.logoModel.ready)) {
      if (status) status.textContent = 'ローカルのロゴモデルが未準備のためプレビューできません。';
      updateLogoPreviewControls();
      return;
    }
    const candidateId = String(candidate.id ?? state.activeCandidateIndex);
    const profiles = selectedLogoProfiles();
    const token = ++state.logoPreviewToken;
    const context = { candidateId, logoRemoval: true, profiles: profiles.slice() };
    state.logoPreviewContext = context;
    state.logoPreviewData = null;
    if (status) status.textContent = 'ロゴ除去プレビューを生成中…';
    const result = document.getElementById('logoPreviewResult');
    if (result) result.classList.add('hidden');
    setBusy(button, true);
    try {
      const data = await api(`/api/jobs/${encodeURIComponent(state.job.id)}/candidates/${encodeURIComponent(candidateId)}/logo-preview`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ logo_removal: true, logo_profiles: profiles }),
      });
      if (token !== state.logoPreviewToken || !sameLogoPreviewContext(context) || !isModalCandidate(candidateId)) return;
      state.logoPreviewData = data || {};
      renderLogoPreviewResult(candidate, data || {});
    } catch (error) {
      if (token === state.logoPreviewToken && sameLogoPreviewContext(context)) {
        if (status) status.textContent = `ロゴ除去プレビューに失敗しました: ${error.message}`;
        showNotice(`ロゴ除去プレビューに失敗しました: ${error.message}`, 'error');
      }
    } finally {
      if (token === state.logoPreviewToken && button) {
        setBusy(button, false);
        updateLogoPreviewControls();
      }
    }
  }

  function sameLogoPreviewContext(context) {
    if (!context || !state.logoPreviewContext) return false;
    return context.candidateId === state.logoPreviewContext.candidateId
      && context.logoRemoval === state.logoPreviewContext.logoRemoval
      && context.profiles.join('\u0000') === state.logoPreviewContext.profiles.join('\u0000');
  }

  function isModalCandidate(candidateId) {
    const pane = els.compareContent.querySelector('.single-preview-pane');
    return Boolean(pane && pane.dataset.candidateId === String(candidateId));
  }

  function renderLogoPreviewResult(candidate, data) {
    const result = document.getElementById('logoPreviewResult');
    const statusNode = document.getElementById('logoPreviewStatus');
    const originalNode = document.getElementById('logoPreviewOriginal');
    const cleanNode = document.getElementById('logoPreviewClean');
    const cleanLabel = document.getElementById('logoPreviewCleanLabel');
    const cleanPlaceholder = document.getElementById('logoPreviewCleanPlaceholder');
    const metaNode = document.getElementById('logoPreviewMeta');
    const warningNode = document.getElementById('logoPreviewWarnings');
    if (!result || !statusNode || !originalNode || !cleanNode || !metaNode || !warningNode) return;
    const previewUrl = data && (data.preview_url || data.cleaned_url || data.output_url || data.url);
    const metadata = data && data.metadata && typeof data.metadata === 'object' ? data.metadata : {};
    const cleanup = metadata.logo_cleanup && typeof metadata.logo_cleanup === 'object' ? metadata.logo_cleanup : (metadata.logoRemoval && typeof metadata.logoRemoval === 'object' ? metadata.logoRemoval : metadata);
    const cleanupStatus = String(firstLogoValue(cleanup.status, metadata.status, data && data.status) || '').toLowerCase();
    const cleanupReason = String(firstLogoValue(cleanup.reason, metadata.reason, data && data.reason) || '').toLowerCase();
    const detected = firstLogoValue(cleanup.detected_profile, cleanup.detectedProfile, cleanup.profile_hits, cleanup.profile, metadata.detected_profile, data && data.detected_profile, data && data.detectedProfile);
    const applied = logoBooleanValue(cleanup.applied, metadata.applied, data && data.applied) ?? ['applied', 'processed', 'cleaned', 'ok', 'done', 'success'].includes(cleanupStatus);
    const noRegisteredProfiles = cleanupStatus === 'off' && /(no\s+(?:registered\s+)?(?:logo\s+)?profiles?|registered\s+(?:logo\s+)?profiles?\s+(?:not\s+found|missing|empty)|未登録)/i.test(cleanupReason);
    const skipped = logoBooleanValue(cleanup.skipped, metadata.skipped, data && data.skipped) === true || ['skipped', 'no_hit', 'no-hit', 'unknown', 'disabled', 'off', 'error'].includes(cleanupStatus);
    const outcome = noRegisteredProfiles ? '未登録のため未加工' : applied === true ? '適用済み' : skipped ? 'スキップ' : '適用状態不明';
    const reasonCopy = logoReasonCopy(cleanupReason, cleanupStatus, applied, skipped, noRegisteredProfiles);
    let warnings = collectLogoWarnings(data, metadata, cleanup, { includeReason: applied !== true });
    if (applied === true) warnings = warnings.filter((item) => !/(registered\s+(?:logo\s+)?matched|logo\s+matched)/i.test(String(item)));
    if (reasonCopy && applied !== true && cleanupReason) warnings = warnings.filter((item) => !String(item).toLowerCase().includes(cleanupReason));
    if (reasonCopy && applied !== true && !warnings.includes(reasonCopy)) warnings.unshift(reasonCopy);
    const appliedReason = applied === true ? reasonCopy : '';
    const detectedText = detected ? `検出プロファイル: ${escapeHTML(logoProfileLabel(detected))}` : '検出プロファイル: なし';
    statusNode.textContent = `ロゴ除去: ${outcome}${detected ? ` · ${logoProfileLabel(detected)}` : ''}${appliedReason ? ` · ${appliedReason}` : ''}`;
    originalNode.src = candidate.preview_url || candidate.crop_url || '';
    if (previewUrl && applied === true) {
      cleanNode.src = String(previewUrl);
      cleanNode.classList.remove('hidden');
      if (cleanLabel) cleanLabel.textContent = 'AI補完後（ロゴ除去）';
      if (cleanPlaceholder) cleanPlaceholder.classList.add('hidden');
    } else {
      cleanNode.removeAttribute('src');
      cleanNode.classList.add('hidden');
      if (cleanLabel) cleanLabel.textContent = noRegisteredProfiles ? 'AI補完後（未加工）' : skipped ? 'AI補完後（スキップ）' : 'AI補完後（未確認）';
      if (cleanPlaceholder) {
        cleanPlaceholder.textContent = noRegisteredProfiles ? 'ロゴ未登録のため元フレームを保持しました' : '元フレームを保持しました';
        cleanPlaceholder.classList.remove('hidden');
      }
    }
    metaNode.innerHTML = `<span>${detectedText}</span><span>状態: ${escapeHTML(outcome)}</span>`;
    warningNode.innerHTML = warnings.length ? `<strong>注意</strong> ${escapeHTML(warnings.join(' · '))}` : (!previewUrl ? 'プレビュー画像が返されませんでした。適用状態は不明です。' : (noRegisteredProfiles ? 'ロゴ未登録のため加工していません。元フレームを保持しました。' : (skipped ? 'ロゴ除去はスキップされ、元フレームを保持しました。' : '')));
    warningNode.classList.toggle('hidden', !warnings.length && Boolean(previewUrl) && !skipped);
    result.classList.remove('hidden');
    if (!previewUrl) statusNode.textContent += ' · プレビュー画像なし';
  }

  function collectLogoWarnings(data, metadata, cleanup = {}, options = {}) {
    const values = [];
    [data && data.warnings, metadata && metadata.warnings, cleanup && cleanup.warnings, cleanup && cleanup.skipped_profiles, data && data.logo_warnings, metadata && metadata.logo_warnings].forEach((raw) => {
      if (Array.isArray(raw)) raw.forEach((item) => { if (item !== null && item !== undefined && String(item).trim()) values.push(String(item)); });
      else if (raw !== null && raw !== undefined && String(raw).trim()) values.push(String(raw));
    });
    if (options.includeReason && cleanup && cleanup.reason && String(cleanup.reason).trim()) values.push(String(cleanup.reason));
    return [...new Set(values.filter((item) => !/(registered\s+(?:logo\s+)?matched|logo\s+matched)/i.test(String(item))))];
  }

  function logoReasonCopy(reason, status, applied, skipped, noRegisteredProfiles) {
    const value = String(reason || '').toLowerCase();
    if (noRegisteredProfiles || (['off', 'skipped'].includes(status) && /(no\s+(?:registered\s+)?(?:logo\s+)?profiles?|未登録)/i.test(value))) return '未登録のため未加工';
    if (!applied && (skipped || ['no_hit', 'no-hit'].includes(status)) && /(no\s+(?:registered\s+)?(?:logo\s+)?matched|no\s+logo\s+matched|no\s+hit|not\s+found|一致しない|見つからない)/i.test(value)) return '登録済みロゴが見つからないため、元フレームを保持しました';
    if (applied && /(registered\s+(?:logo\s+)?matched|logo\s+matched|matched|一致)/i.test(value)) return '登録済みロゴを検出しました';
    return '';
  }

  function firstLogoValue(...values) {
    return values.find((value) => value !== undefined && value !== null && String(value).trim() !== '');
  }

  function logoBooleanValue(...values) {
    const value = firstLogoValue(...values);
    if (value === undefined) return null;
    if (typeof value === 'boolean') return value;
    if (typeof value === 'number') return value !== 0;
    const text = String(value).trim().toLowerCase();
    if (['true', 'yes', 'applied', 'processed', 'done', 'success'].includes(text)) return true;
    if (['false', 'no', 'skipped', 'skip', 'off', 'error'].includes(text)) return false;
    return null;
  }

  function logoProfileLabel(value) {
    if (Array.isArray(value)) {
      const labels = value.map((item) => logoProfileLabel(item)).filter(Boolean);
      return labels.join('、');
    }
    if (value && typeof value === 'object') {
      const nested = value.id ?? value.profile_id ?? value.name ?? value.label;
      return nested === undefined ? '' : logoProfileLabel(nested);
    }
    const id = String(value);
    const profile = state.logoProfiles.find((item) => item.id === id);
    return profile ? `${profile.name}（${profile.id}）` : id;
  }

  function openCompare() {
    const selected = state.candidates.filter((candidate, index) => state.selectedIds.has(String(candidate.id ?? index))).slice(0, 2);
    if (selected.length !== 2) return;
    els.compareKicker.textContent = 'COMPARE';
    els.compareTitle.textContent = '2枚を比較';
    els.compareContent.className = 'compare-content';
    els.compareContent.innerHTML = selected.map((candidate) => `<div class="compare-pane"><img src="${escapeAttribute(candidate.preview_url || candidate.crop_url || '')}" alt="候補 ${escapeAttribute(formatTime(candidate.time))}"><h3>${escapeHTML(formatTime(candidate.time))} · スコア ${formatPercent(candidate.score)}</h3><p>${escapeHTML(evidenceLine(candidate))}</p><img class="evidence-crop" src="${escapeAttribute(candidate.crop_url || candidate.preview_url || '')}" alt="対象 crop"></div>`).join('');
    openModal(els.compareModal);
  }

  async function exportSelected() {
    if (!state.job || !state.job.id || !state.selectedIds.size) return;
    setBusy(els.exportButton, true);
    try {
      const logoRemoval = logoRemovalEnabled();
      const data = await api(`/api/jobs/${encodeURIComponent(state.job.id)}/export`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ candidate_ids: [...state.selectedIds], logo_removal: logoRemoval, logo_profiles: logoRemoval ? selectedLogoProfiles() : [] }),
      });
      const summary = summarizeExport(data || {}, logoRemoval);
      showNotice(summary.message, summary.warning ? 'warning' : 'success', 9000);
    } catch (error) { showNotice(`書き出しに失敗しました: ${error.message}`, 'error'); }
    finally { setBusy(els.exportButton, false); }
  }

  function summarizeExport(data, logoRemoval) {
    const metadata = data && data.metadata && typeof data.metadata === 'object' ? data.metadata : {};
    const primaryFiles = firstExportArray(data.files, data.exported_files, data.output_files);
    const originalFiles = firstExportArray(data.original_files, data.originals, data.original_frame_files, data.source_files, metadata.original_files, metadata.originals);
    const primaryCount = firstExportCount(data.png_count, data.primary_count, data.exported_count, data.output_count, metadata.png_count, metadata.primary_count) ?? countPngFiles(primaryFiles);
    const originalCount = firstExportCount(data.original_count, data.originals_count, data.original_frame_count, metadata.original_count, metadata.originals_count) ?? originalFiles.length;
    const warnings = collectExportWarnings(data, metadata);
    const parts = [`PNG ${primaryCount}枚`, `元フレーム ${originalCount}枚`];
    if (!logoRemoval) parts.push('ロゴ除去なし');
    if (warnings.length) parts.push(`注意: ${warnings.join(' · ')}`);
    if (data && data.directory) parts.push(`保存先: ${data.directory}`);
    return { message: `書き出し完了 · ${parts.join(' · ')}`, warning: warnings.length > 0 };
  }

  function firstExportArray(...values) {
    return values.find((value) => Array.isArray(value)) || [];
  }

  function firstExportCount(...values) {
    const value = values.find((candidate) => candidate !== undefined && candidate !== null && candidate !== '');
    if (value === undefined) return null;
    const count = Number(value);
    return Number.isFinite(count) && count >= 0 ? Math.round(count) : null;
  }

  function countPngFiles(files) {
    return files.filter((file) => {
      const path = typeof file === 'string' ? file : file && (file.path || file.name || file.url || '');
      return String(path).toLowerCase().endsWith('.png');
    }).length;
  }

  function collectExportWarnings(data, metadata) {
    const values = [];
    [data && data.logo_warnings, data && data.skipped_logos, data && data.warnings, metadata && metadata.logo_warnings, metadata && metadata.skipped_logos, metadata && metadata.warnings].forEach((raw) => {
      if (Array.isArray(raw)) raw.forEach((item) => { if (item !== null && item !== undefined && String(item).trim()) values.push(String(item)); });
      else if (raw !== null && raw !== undefined && String(raw).trim()) values.push(String(raw));
    });
    if (data && data.logo_skipped === true && !values.length) values.push('ロゴ除去はスキップされ、元フレームを保存しました');
    return [...new Set(values.filter((item) => !/(registered\s+(?:logo\s+)?matched|logo\s+matched)/i.test(String(item))))];
  }

  function openModal(modal) { modal.classList.remove('hidden'); document.body.classList.add('modal-open'); const close = modal.querySelector('.close-button'); close?.focus(); }
  function closeModals() { invalidateLogoPreview(); els.compareModal.classList.add('hidden'); els.shortcutModal.classList.add('hidden'); document.body.classList.remove('modal-open'); }

  function handleGlobalKeydown(event) {
    if (event.key === 'Escape') { if (!els.compareModal.classList.contains('hidden') || !els.shortcutModal.classList.contains('hidden')) closeModals(); else hideSuggestions(); return; }
    if (!els.compareModal.classList.contains('hidden') && els.compareContent.classList.contains('single-preview') && (event.key === 'ArrowLeft' || event.key === 'ArrowRight')) { event.preventDefault(); stepCandidate(event.key === 'ArrowLeft' ? -1 : 1); return; }
    if (event.target.matches('input, textarea, select, button')) return;
    if (event.key === 'ArrowLeft') { event.preventDefault(); stepCandidate(-1); }
    if (event.key === 'ArrowRight') { event.preventDefault(); stepCandidate(1); }
    if (event.code === 'Space' && state.activeCandidateIndex >= 0) { event.preventDefault(); const candidate = state.candidates[state.activeCandidateIndex]; const id = String(candidate.id ?? state.activeCandidateIndex); toggleCandidate(id, !state.selectedIds.has(id)); const card = document.querySelector(`.candidate-card[data-id="${cssEscape(id)}"] input`); if (card) card.checked = state.selectedIds.has(id); }
  }

  function updateSummary() { els.summaryReference.textContent = `${state.references.length}枚`; els.summaryTags.textContent = `${state.wanted.length + state.avoided.length}件`; }
  function updateCaptureAvailability() {
    els.captureReference.disabled = !state.currentVideo || state.videoLoading || state.captureLoading;
    els.captureReference.setAttribute('aria-busy', state.videoLoading || state.captureLoading ? 'true' : 'false');
  }

  function isConditionTag(tag) {
    if (!tag || typeof tag !== 'object') return false;
    const category = normalizeTagName(tag.category ?? tag.group ?? tag.scopeCategory ?? '');
    if (!category) return false;
    const words = category.replace(/[/:|]+/g, ' ').split(/\s+/).filter(Boolean);
    return words.some((word) => ['general', 'expression', 'pose', 'composition'].includes(word));
  }
  function rememberVocabularyTags(tags) {
    tags.forEach((tag) => {
      const name = tag && (tag.name || tag.tag);
      if (!name) return;
      const canonical = String(name);
      if (isConditionTag(tag)) state.vocabulary.set(canonical, tag);
      else state.blockedTags.add(normalizeTagName(canonical));
    });
  }
  function updateCancelControl() {
    const active = Boolean(state.job && ['queued', 'running'].includes(state.job.status));
    const requested = Boolean(state.job && state.job.cancelRequested);
    els.cancelJob.disabled = !active || requested;
    els.cancelJob.textContent = requested ? 'キャンセル要求済み' : '解析をキャンセル';
    els.cancelJob.setAttribute('aria-busy', requested ? 'true' : 'false');
  }
  function updateWeightTotal() { const sum = getNumber(els.identityWeight, .35) + getNumber(els.conditionWeight, .3) + getNumber(els.qualityWeight, .35); els.weightTotal.textContent = sum.toFixed(2); els.weightTotal.style.color = Math.abs(sum - 1) < .001 ? 'var(--mint)' : 'var(--amber)'; }
  function setBusy(node, busy) { if (!node) return; node.disabled = Boolean(busy); node.setAttribute('aria-busy', busy ? 'true' : 'false'); }
  function getNumber(input, fallback) { const value = Number(input && input.value); return Number.isFinite(value) ? value : fallback; }
  function statusLabel(status) { return ({ queued: 'キュー待機中…', running: 'フレームを解析中…', done: '完了', cancelled: 'キャンセル済み', error: 'エラー' })[status] || '解析中…'; }
  const shortageReasonCopy = {
    no_identity_match: '対象に一致する候補が見つかりませんでした',
    tag_gate: 'タグ条件に合う候補が不足しています',
    no_condition_match: '条件に合う候補が不足しています',
    tag_or_quality_gate: 'タグ・画質条件に合う候補が不足しています',
    pixel_deduplication: '重複を除いた候補が不足しています',
    sample_budget: '探索上限のため候補が不足しています',
    insufficient_matching_frames: '一致するフレームが不足しています',
  };
  function shortageMessage(stats, count) {
    if (!count) return '条件に合う候補が見つかりませんでした。タグ閾値や画質下限を下げて再試行してください。';
    if (stats && (stats.shortage || stats.shortage_warning || stats.shortage_reason)) {
      return humanizeShortage(stats.shortage_warning || stats.shortage_reason, stats.shortage);
    }
    if (count < 5) return `候補が${count}件です。必要に応じて探索区間や閾値を調整してください。`;
    return '';
  }
  function humanizeWarning(value) {
    const text = String(value ?? '');
    const reason = Object.keys(shortageReasonCopy).find((key) => text.toLowerCase().includes(key));
    if (!reason) return text;
    const countMatch = text.match(/(\d+)\s*(?:candidate|候補)/i);
    return humanizeShortage(reason, countMatch ? Number(countMatch[1]) : null);
  }
  function humanizeShortage(reasonOrWarning, count) {
    const text = String(reasonOrWarning ?? '');
    const reason = Object.keys(shortageReasonCopy).find((key) => text.toLowerCase().includes(key)) || text;
    const copy = shortageReasonCopy[reason] || '候補が不足しています';
    const parsedCount = Number.isFinite(Number(count)) && Number(count) > 0 ? Number(count) : ((text.match(/(\d+)\s*(?:candidate|候補)/i) || [])[1] ? Number((text.match(/(\d+)\s*(?:candidate|候補)/i) || [])[1]) : 0);
    return parsedCount > 0 ? `${copy}（${parsedCount}件不足）` : copy;
  }
  function evidenceLine(candidate) { return `同一性 ${formatPercent(candidate.identity)} · 画質 ${formatPercent(candidate.quality)} · 状態 ${formatPercent(candidate.condition)}`; }
  function tagSnippet(tags) {
    const flattened = flattenCandidateTags(tags).sort((left, right) => right.confidence - left.confidence).slice(0, 3);
    return flattened.map((item) => `<span title="${escapeAttribute(`${item.scope === 'subject' ? '対象' : 'フレーム'}: ${item.tag}`)}"><b>${item.scope === 'subject' ? '対象' : 'フレーム'}</b> ${escapeHTML(item.tag)} ${formatPercent(item.confidence)}</span>`).join('');
  }
  function tagEvidence(tags) { const snippet = tagSnippet(tags); return snippet ? `<div class="evidence-tag-line">${snippet}</div>` : ''; }

  // Model taggers expose category maps (general/character/tags), while the
  // pipeline wraps those maps once more under subject/frame. Keep the UI
  // tolerant of both shapes and discard non-numeric values before rendering.
  function flattenCandidateTags(tags) {
    if (!tags || typeof tags !== 'object' || Array.isArray(tags)) return [];
    const output = [];
    const hasScopes = Object.prototype.hasOwnProperty.call(tags, 'subject') || Object.prototype.hasOwnProperty.call(tags, 'frame');
    if (hasScopes) {
      flattenTagValue(tags.subject, 'subject', output, 0);
      flattenTagValue(tags.frame, 'frame', output, 0);
    } else {
      flattenTagValue(tags, 'frame', output, 0);
    }
    const deduped = new Map();
    output.forEach((item) => {
      const key = `${item.scope}:${item.tag.toLocaleLowerCase()}`;
      const previous = deduped.get(key);
      if (!previous || item.confidence > previous.confidence) deduped.set(key, item);
    });
    return [...deduped.values()];
  }

  function flattenTagValue(value, scope, output, depth) {
    if (depth > 5 || value === null || value === undefined) return;
    if (Array.isArray(value)) {
      value.forEach((item) => {
        if (!item || typeof item !== 'object') return;
        const name = item.tag ?? item.label ?? item.name;
        const confidence = item.score ?? item.confidence ?? item.probability;
        if (name !== undefined && numericConfidence(confidence) !== null) {
          output.push({ tag: String(name), confidence: numericConfidence(confidence), scope });
        } else flattenTagValue(item, scope, output, depth + 1);
      });
      return;
    }
    if (typeof value !== 'object') return;
    Object.entries(value).forEach(([key, raw]) => {
      const confidence = numericConfidence(raw);
      if (confidence !== null) output.push({ tag: String(key), confidence, scope });
      else if (raw && typeof raw === 'object') flattenTagValue(raw, scope, output, depth + 1);
    });
  }

  function numericConfidence(value) {
    if (typeof value === 'boolean' || typeof value === 'object' || value === null || value === undefined || value === '') return null;
    const number = Number(value);
    return Number.isFinite(number) ? clamp(number, 0, 1) : null;
  }
  function formatVideoInfo(info) { const parts = []; if (info.width && info.height) parts.push(`${info.width} × ${info.height}`); if (info.duration !== undefined) parts.push(formatTime(info.duration)); if (info.fps) parts.push(`${formatNumber(info.fps)} fps`); return parts.join(' · ') || 'メタデータ待機中'; }
  function formatTime(value) { const seconds = Math.max(0, Number(value) || 0); const minutes = Math.floor(seconds / 60); const rest = seconds - minutes * 60; return `${String(minutes).padStart(2, '0')}:${rest.toFixed(1).padStart(4, '0')}`; }
  function formatPercent(value) { const number = Number(value); return Number.isFinite(number) ? `${Math.round(number * 100)}%` : '—'; }
  function formatNumber(value) { const number = Number(value); return Number.isFinite(number) ? number.toFixed(3) : '—'; }
  function formatScope(scope) { return scope === 'subject' ? '対象' : scope === 'frame' ? 'フレーム' : 'タグ'; }
  function normalizeBox(box) { if (!Array.isArray(box) || box.length < 4 || box.some((value) => !Number.isFinite(Number(value)))) return null; const x0 = clamp(Number(box[0]), 0, 1); const y0 = clamp(Number(box[1]), 0, 1); const x1 = clamp(Number(box[2]), 0, 1); const y1 = clamp(Number(box[3]), 0, 1); return [Math.min(x0, x1), Math.min(y0, y1), Math.max(x0, x1), Math.max(y0, y1)]; }
  function normalizeTagName(value) { return String(value ?? '').trim().toLocaleLowerCase().replace(/[_-]+/g, ' ').replace(/\s+/g, ' '); }
  function vocabularyCanonical(value) {
    const needle = normalizeTagName(value);
    if (!needle) return '';
    const match = [...state.vocabulary.keys()].find((name) => normalizeTagName(name) === needle);
    return match || '';
  }
  function reverseAlias(canonical, query) {
    const target = normalizeTagName(canonical);
    const queryKey = normalizeTagName(query);
    const match = Object.entries(state.aliases).find(([alias, value]) => normalizeTagName(value) === target && (!queryKey || normalizeTagName(alias).includes(queryKey)));
    return match ? match[0] : '';
  }
  function resolveAlias(query) {
    const needle = normalizeTagName(query);
    const exact = Object.entries(state.aliases).find(([alias]) => normalizeTagName(alias) === needle);
    if (!exact) return '';
    const target = String(exact[1]).trim();
    if (state.blockedTags.has(normalizeTagName(target))) return '';
    return vocabularyCanonical(target) || target;
  }
  function isKnownTag(name) {
    const value = String(name || '').trim();
    if (state.blockedTags.has(normalizeTagName(value))) return false;
    if (vocabularyCanonical(value)) return true;
    const needle = normalizeTagName(value);
    return Boolean(needle && Object.values(state.aliases).some((tag) => !state.blockedTags.has(normalizeTagName(tag)) && normalizeTagName(tag) === needle));
  }
  function resolveKnownTag(name) {
    const value = String(name || '').trim();
    const direct = vocabularyCanonical(value);
    if (direct) return direct;
    const alias = resolveAlias(value);
    if (alias && isKnownTag(alias)) return vocabularyCanonical(alias) || alias;
    return isKnownTag(value) ? value : '';
  }
  function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
  function basename(path) { return String(path || '').split(/[\\/]/).pop() || '動画'; }
  function escapeHTML(value) { return String(value ?? '').replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char])); }
  function escapeAttribute(value) { return escapeHTML(value); }
  function cssEscape(value) { return String(value).replace(/[^a-zA-Z0-9_-]/g, '\\$&'); }

  function showToast(message) { els.toast.textContent = message; els.toast.classList.add('show'); clearTimeout(state.noticeTimer); state.noticeTimer = setTimeout(() => els.toast.classList.remove('show'), 3500); }
  function clearGlobalNotice() { els.globalNotice.classList.add('hidden'); }
  function showNotice(message, type = 'warning', duration = 6000) { els.globalNotice.textContent = message; els.globalNotice.className = `global-notice ${type === 'error' ? 'error' : type === 'success' ? 'success' : ''}`; if (duration > 0) { clearTimeout(state.noticeTimer); state.noticeTimer = setTimeout(() => els.globalNotice.classList.add('hidden'), duration); } }

  async function api(path, options = {}) {
    const response = await fetch(path, options);
    const text = await response.text();
    let data = null;
    try { data = text ? JSON.parse(text) : {}; } catch (_) { data = { detail: text }; }
    if (!response.ok) throw new Error((data && (data.detail || data.message || data.error)) || `HTTP ${response.status}`);
    return data;
  }
})();
