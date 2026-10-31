# Edit Helper - Redesign v2 (light + dark, anti-slop pass)

## How to apply
Two files, no other changes:
1. Replace `gui/app.py` with this `app.py`.
2. Replace `.streamlit/config.toml` with this `config.toml` (neutral light base +
   keeps your 10GB upload limit; the app ships its own theme system).
Run `run_app.bat` as usual. Nothing in your repo was changed by Instinct; no pushes,
no deployments. Your data, projects, keys and settings are untouched.

## What changed in v2 (your feedback: no AI slop, modern, light AND dark)
Design research actually performed (screenshots studied, not just named):
- superhero.io - restraint: one accent, confident type, one clear CTA.
- navbar.gallery / cta.gallery - tool shells: left rail, functional density.
- pricingpages.design - light canvases with hierarchy from type + whitespace.
- footer.design - typographic categorization, quiet structure.
- Plus published "AI slop" pattern catalogs (gradient/glow/default-card tells)
  and professional editing tools (Descript, Frame.io) for workspace conventions.

Decisions that came out of that:
- Two full themes with an Appearance toggle in the sidebar (Light / Dark).
  The choice persists across reloads via the ?theme= URL parameter.
- No gradients, no glows, no lift-on-hover, no glass. Hierarchy comes from
  typography, hairline rules and spacing - not from cards everywhere.
- JetBrains Mono for timecodes, durations, step numbers, meta data - the
  visual DNA of real editing tools (Premiere/DaVinci/Frame.io).
- Orange is functional, not decorative: primary actions, active states,
  timecode chips. Solid fills only. Light theme uses a deeper burnt orange
  (#C2530C) for AA contrast on white; dark uses #FF8A3D on near-black.
- Landing feature list is a hairline-ruled row, not a card grid.
- Step tracker is a flat numbered strip, not gradient pills.
- Scene plan rows keep Streamlit expanders (backend wiring) but restyled:
  transparent headers, readable translation blocks, squared prompt areas.

## Verified in a real run (Streamlit 1.65, ML backends stubbed)
- Desktop 1440px AND mobile 390px, in BOTH themes (toggle clicked in a real
  browser; canvas color verified programmatically after the flip).
- Empty landing (light + mobile), seeded project, scene plan, results gallery.
- Fixed during verification: sidebar stayed dark in light theme (hardcoded
  color), primary button kept an old orange gradient, translation text was
  low-contrast, expander headers flashed white in dark mode, the info alert
  ignored theme tokens (replaced with a themed note box), disabled buttons
  were unreadable. All confirmed fixed in final screenshots.

## Honest limits in the copy (unchanged policy)
- No "any language / any length" claims. UI says multilingual including Tamil,
  quality varies by model and clip, and cites the configured 10GB upload cap.

## Models and providers (configurable, nothing hardcoded)
- The redesign does not lock you to a model. Keep provider and model IDs in
  Settings/.env server-side; never in browser code or localStorage.
- Audio understanding and image generation are separate stages: you can keep
  Gemini for Tamil understanding and swap the image provider independently.
- Researched options (official pages, no tests run):
  - Images: gemini-nano-banana-2.1 ~$0.0336 per 1K image
    (ai.google.dev/gemini-api/docs/models/gemini-nano-banana-2.1);
    fal FLUX.1 schnell ~$0.003/MP (fal.ai/models/fal-ai/flux/schnell);
    gpt-image-1-mini ~$0.011 per 1024px image
    (developers.openai.com/api/docs/models/gpt-image-1-mini).
    Prices are from those pages today; treat as volatile, quality untested.
  - Understanding/planning: a Flash-class Gemini with audio input.
- Proposal for a follow-up (NOT implemented): estimated cost before Generate,
  per-image regeneration, provider dropdown for image generation.

## Safety/security inspection of the existing backend (unchanged, awareness)
- API keys are stored in PLAINTEXT in the local sqlite db (data/edit_helper.db).
  Fine for local single-user; do not share that file.
- No authentication layer. Keep it on localhost; never --server.address 0.0.0.0.
- "Local file path" input reads any file path on your disk by design.
- ffmpeg is invoked with argument lists (no shell interpolation seen).

## Screenshot honesty
Screenshots show the redesigned app actually running with stubbed ML backends.
The sample project, Tamil transcript moments and gallery images are test data;
the 3 gallery images are AI-generated sample assets made for this review, NOT
output from the app's Gemini pipeline (no paid API calls were made).

## Independent audit repair round (Oct 29, 2026, late)

An independent reviewer agent ran the seeded app (7 library resources) on desktop
and mobile in both themes and scored components. Repairs applied to the local
review implementation (this app.py), then re-verified in rendered pixels:

- Library Copy/Delete white buttons in dark (2/10): root cause was buttons wrapped
  in stTooltipHoverTarget, so child selectors missed them and they fell back to
  Streamlit's compiled light theme. Rewritten as descendant selectors.
- Library badges (Pinterest/Pexels) and code chips: per-theme --badge-red /
  --badge-green tokens with calm rgba tints; chips pinned to surface-3/muted.
- Filter selectbox value nearly invisible (3/10): this Streamlit version renders
  selectboxes as react-aria combobox inputs, not baseweb selects; value, border,
  open button and dropdown options re-pinned to theme tokens.
- New Project form placeholders/borders (4/10) and Stock pipeline sliders,
  number-input steppers (4/10): inputs, placeholders, slider values/labels,
  stepper buttons pinned to theme tokens; default input borders calmed to
  --border with --border-strong on hover and accent on focus.
- Step tracker: now marks the first incomplete step active (steps 2-3 done when
  analysis exists, 4 done when images exist) instead of staying on step 1;
  inactive labels raised from --faint to --muted (light ~3.1:1 finding).
- Misleading copy fixed: "(no size limit)" -> "(up to 10GB, set in .streamlit
  config)"; "any size" -> "up to 10GB".
- Mobile sidebar: initial_sidebar_state "expanded" -> "auto" so the sidebar no
  longer covers the page on phones.
- Activity chat input: white outer panel and invisible placeholder in dark
  (2/10) fixed (outer container, inner wrapper, textarea, placeholder, submit
  button); avatars given themed tile + border; collapse chevron contrasted.
- Global Settings: password eye-button white leak fixed (transparent bg, muted
  icon); tooltip/help icons raised to --muted.
- Uploaded-file chip (2/10 dark): chip is data-testid stFileChip in this
  Streamlit version (not stFileUploaderFile); chip tile, filename, size, delete
  and "+" buttons themed in both modes.
- Keyboard focus (2/10): consistent 2px theme-accent :focus-visible outline on
  all interactive elements, replacing Streamlit's leaked faint primary-color
  shadow ring; input root containers get accent focus-within borders.

Honest limits of this verification:
- The open selectbox dropdown would not open under synthetic input in headless
  Chrome; listbox/option CSS is in place but the open state needs the reviewer's
  re-check.
- Focused-input accent ring verified orange via keyboard Tab in dark; the
  focus-within root-element rule was added after the last full screenshot pass,
  so a focused Settings field should be re-checked by the reviewer.
- All evidence is from the stub backend with seeded sample data. No paid APIs
  were called and no user media was processed.

## Final low-score repair round (v6)

Second reviewer pass confirmed the major fixes and scored the remaining cosmetic
issues; all were repaired and re-verified in rendered pixels:

- R1 sidebar section labels (PROJECTS/APPEARANCE) and brand tagline: --faint ->
  --muted (light 3.28:1 -> ~6:1, dark 4.35:1 -> ~7.7:1).
- R2 small light eyebrows (.section-eyebrow) --faint -> --muted; done-state step
  numerals --accent -> --accent-2 (light 4.43:1 -> ~5.9:1).
- R3 slider values: measured values already render in --text at max contrast;
  the drag bubble (stThumbValue) is now themed surface-3/text so the value can
  never sit white-on-accent while dragging.
- R4 type chips: 10.5px muted -> ~11px --text at weight 500 on surface-3.
- R5 Find-stock placeholder now clearly distinct from entered text: short
  italic placeholder ("Describe the visual you need") with the long example
  moved into a helper caption below the input, which also notes Tamil/Hindi
  translation.
- R7 help '?' icons: white discs came from Streamlit's higher-specificity
  !important button styles; re-pinned with higher-specificity selectors -
  transparent background, muted outline icon, hover raises to text/surface-3.
- R6: avatars unverified by the reviewer (headless emoji fonts); no defect
  proven, tiles are themed. Empty chat send arrow is the intentional disabled
  state. R8 disabled Analyze: cosmetic only, left as-is per reviewer note.

Verification: same seeded harness, both themes, desktop + mobile; computed
probes plus pixel inspection of each repaired region.

## v7 targeted repair (reviewer-supplied exact selectors)

Reviewer matched DOM nodes with matches()/computed styles; fixes target those
exact nodes:

- Copy/Delete pill regression: the help-icon reset was hitting every button
  wrapped in stTooltipIcon/stTooltipHoverTarget (all help= buttons) at equal
  specificity but later in the sheet. Reset now scoped to
  button:not([kind]) - actual icon-only help buttons - so content buttons keep
  their pills.
- Candidates help disc: the number-input rule `button svg { fill: --text }`
  was catching the help icon in the widget label. Fill rule scoped to
  stNumberInputContainer steppers; explicit fill:none/stroke:muted override for
  the label help icon.
- Slider value: actual node is stSliderThumbValue > stMarkdownContainer > p
  (was targeting stThumbValue). Value now renders --text on surface-3 bubble;
  was burnt orange (compiled light-theme primary).
- proj-meta and the COVERAGE ACROSS YOUR VIDEO label: --faint -> --muted.
- img-chip "Place at" tags and hero 01-04 numerals: --accent -> --accent-2
  (light ~4.4 -> ~5.9:1). Note: hero large text has a lower AA threshold
  (3:1); only the small numerals/chips were changed.

Verified via computed probes (both themes) plus pixel inspection: button pills
rgb(33,36,42)+border dark / rgb(244,243,239)+border light, help icon
transparent/muted with fill none, thumb value text in both themes, chips and
numerals rgb(158,66,7) in light.

## v8 final targeted repair

- Inactive step-strip numerals (Create, pre-analysis): override block colored
  them --faint (3.14:1 in light at 11px mono). Now --muted. Verified on the
  running seeded app in the exact pre-analysis state (project open, no
  analysis): light rgb(91,94,101) ~5.9:1, dark rgb(162,168,177) ~7:1, both at
  11px mono. Active step unchanged (accent fill + accent-ink numeral), done
  steps unchanged (accent-2). Pixel shots: v8-light-create-steps-todo.png,
  v8-dark-create-steps-todo.png.

## v9 expanded-state repair (toasts, tooltip, dropzone hover, code tokens)

- Toasts (Copied!/Saved): compiled light theme leaked white boxes with light
  grey text (~2.4:1 dark). stToast/stToastContainer/[data-baseweb=toast] now
  themed: dark bg rgb(33,36,42) + border + text rgb(236,237,239); light bg
  rgb(244,243,239) + text rgb(27,28,31). Verified by clicking Copy live.
- Copy tooltip: stTooltipContent/baseweb tooltip+popover wrapper now surface-3
  + --text + themed border. Verified dark bg rgb(42,46,54) text rgb(236,237,239);
  light bg rgb(233,231,225) text rgb(27,28,31).
- Upload dropzone hover/drag/focus-within: was white flash in dark (2.14:1).
  Base + :hover + :focus-within all force surface-2 with !important (drag state
  only swaps an emotion class, so the base !important covers it), text forced
  --muted, hover border accent. Verified on hover: bg rgb(33,36,42), text
  rgb(162,168,177).
- Agent Memory JSON code tokens (Prism .token.* classes in this Streamlit):
  base pre surface-2 + --text; keys/numbers --accent-2, strings --green,
  punctuation/booleans --muted. Verified live both themes, all >= 4.5:1
  (dark: punctuation 6.8, keys 9+, strings 8.1; light: punctuation 5.9,
  keys 5.85, strings 4.7).

## v10 drag-active + media-type caption

- Dropzone drag-active: a real dragenter/dragover with a DataTransfer+File
  swaps the section's emotion class and inserts a "Drag and drop a file here"
  div before the limits line (verified: classes 53atgp->ymc37v, inner
  e12r9z2m3->e12r9z2m1 inserted). Fixes: all inner layers forced transparent
  so the class-swap repaint (rgb(240,242,246)) cannot show; Upload button pill
  re-asserted; limits line hidden structurally during drag
  (> div:not(:only-of-type) ~ div) because Streamlit hides it by color, which
  the muted-text guard defeats; drag instruction set to --text/600. Verified
  mid-drag: dark section rgb(33,36,42), instruction rgb(236,237,239), limits
  hidden; light section rgb(244,243,239), instruction rgb(27,28,31). Idle
  state unchanged (limits visible).
- Find stock media-type caption ("image" beside source badge): was inline
  #9AA3AF (2.44:1 light). Now .media-type-tag = --muted. Verified light
  rgb(91,94,101) ~5.9:1, dark rgb(162,168,177), at 12.8px.

## v11 - navigation redesign (user-approved)

User feedback: the pill tab bar (Create images / Library / ...) felt cramped, the boxed
active tab looked awkward, and tabs disappeared sideways on mobile. Approved direction:
spacious underline navigation on desktop, clearly labelled page picker on mobile.

- Replaced st.tabs with a navigation state (_page) + two synced widgets:
  st.segmented_control (desktop, restyled as an underline bar) and a st.selectbox
  labelled "Go to page" (mobile, CSS swaps them at 700px). Content dispatch changed
  from `with tab_*:` blocks to `if _page == ...:`; page bodies unchanged.
- Desktop bar: 15px/600 labels, 30px gaps, hairline bottom border, active page gets
  full-color text + 3px accent underline (#C2530C light / #FF8A3D dark). No boxes or
  pill backgrounds anywhere. Inactive labels use --muted (5.9:1 light, higher dark).
- Keyboard focus: 2px accent outline on nav buttons preserved (verified computed style).
- Mobile picker: standard selectbox, full width, visible label; verified switching
  pages with a real click at 390px; no horizontal scroll.
- Obsolete CSS removed: pill tab styles and the stTabs scroll-button overrides.
- Verified: desktop light/dark (switch Create images -> Library), mobile light
  (picker open + switch to Library), focus ring, no server warnings.

## v12 - landing page integration (user goal: complete the project)

User set the goal of completing the project end to end; first step was wiring the
approved Toolcraft landing concept into the real app (landing at start, "Try it"
opens the generate page, way back).

- Added an entry gate (session state "entered"): before it, the app renders a
  marketing landing instead of the editor. Ported from the Toolcraft v2 concept:
  fixed dark palette (#0c0b0d, orange #FF8A3D), headline "Every key moment of
  your video, as a picture.", honest subcopy (many languages, quality varies).
- Hero: the concept's canvas waveform animation ported to vanilla JS in a
  components iframe, draw logic unchanged - five labelled sample scenes develop
  at key moments as the playhead passes, "Sample talk, sample pictures" label
  kept, prefers-reduced-motion shows the finished state.
- Try it (orange pill) enters the app: with existing projects it opens the
  newest project straight on Create images (v11 nav stays in sync); with no
  projects it opens the New Project form. "Opens the generate page" caption kept.
- Way back: quiet text button "Back to landing" at the top of the sidebar.
- Landing keeps its fixed dark identity; the editor keeps its own light/dark
  toggle (per the concept author's integration note, the dark landing was the
  reviewed design).
- Verified live: landing desktop + mobile (2x2 hero grid, no hscroll), Try it ->
  Create images, Back -> landing, reduced-motion finished state, orange CTA and
  quiet back button after a specificity fix against the theme's !important rules.
- Not exercised: empty-database first-run path (harness always seeds a project).

## v13 - completion hardening (explicit fallbacks, persistence, recovery, readiness)

Continues the "complete the project" goal: honest fallbacks instead of silent fake
success, transcript persistence, error recovery, and a security readiness pass.

Explicit fallbacks (no more silent fake success)
- Find stock / pipeline cards from the built-in curated catalog now show a SAMPLE
  badge (was FALLBACK) and an inline notice: "No live stock source answered (offline
  or no Pexels key). Cards marked SAMPLE are curated stand-ins - not live search
  results. Add a Pexels key in Global Settings for real results."
- MultimodalEmbedder gained an is_mock property; the pipeline and aesthetic-memory
  sync now warn when similarity scoring runs in deterministic fallback mode
  ("rankings are approximate, not real quality judgments"). Not exercised in the
  review harness (no torch); code path verified by inspection.

Persistence + recovery
- Transcript analysis now saves to project memory (last_analysis) and restores on
  reload - verified: fresh session restores "3 moments found" for the saved source
  file. Selecting a different file still invalidates it.
- Pipeline: transcription returning no beats used to crash (TypeError on len(None));
  now a clean error ("no usable speech... switch the transcription method") and the
  Run button recovers. Verified with a stubbed transcription failure.
- Local-path input: nonexistent path shows "File not found" and the Run button
  re-enables. Verified.

Security readiness assessment (local single-user desktop app)
- API keys are stored unencrypted in the local sqlite (data/app.db). Acceptable for
  a private local app; the settings UI now says so and advises a rotatable key.
  Real fix (OS keyring / encryption at rest) is a larger decision, not done.
- No authentication: the app must stay a local tool. The new landing gate is a UX
  entry view, NOT an access control - do not expose the port or host as-is.
- Local file paths: pipeline reads user-specified local paths as the local user
  (expected for a desktop tool); existence is validated and errors are clean.
- Clipboard copy helper is Windows-specific; harmless no-op elsewhere.

CTA repair (from design review of v12)
- Try it label is now bold dark (#1a0e05, 700) on orange - the theme's inner-text
  rules had overridden the button-level color. "Opens the generate page" sits
  directly under the CTA (10px), adjacent on phone too.

---

## v14 — Fallback honesty, pipeline error states, key storage (2026-10-30)

**Fallback SAMPLE labeling (parent catch).** v13's Find-stock shot showed one fixture card with a PEXELS badge next to SAMPLE-labeled ones. Root cause was harness seed data (a stand-in card marked source='pexels'), not app code - app code labels every `source == "fallback"` card SAMPLE. Harness seeds are now fallback-only, so no fixture can masquerade as a live-provider result. Verified: 2/2 stand-in cards carry SAMPLE badges plus the "not live results" notice; zero PEXELS badges on the page (v14-findstock-sample.png).

**Real bug: silent demo-sample fallback.** Selecting "Local file path" (or "Upload file") with no path/file silently ran the bundled demo sample instead of erroring - the exact silent-fake-success pattern we are removing. Each input method now errors explicitly ("Upload a file first", "Enter a local file path first", "Demo sample is missing"), and the progress bar is cleared on these stops so the final state is unambiguous (v14-pipeline-error.png: "File not found: /nonexistent/does-not-exist.mp4" with Run Pipeline re-enabled).

**API key storage rework.**
- Precedence changed from db-first to: session-only key > environment (.env) > stored db key.
- Global Settings: each key now has "Remember ... on this device" (default ON only if a key is already stored - no silent loss of existing setups; OFF for fresh installs), plus an explicit "Remove stored key" action when a stored key exists. Nothing is deleted automatically.
- Create-page key prompt no longer writes to the db at all - it is session-only, with help text pointing to Global Settings for persistence. (This second write path bypassed the new settings flow entirely; a scripted check caught a fake key persisting through it. After the fix, the same scripted flow leaves zero api_key rows in the db.)
- Caption now states: session keys live in memory only; remembered keys are stored unencrypted in data/app.db; env vars are never written by the app.
- Migration impact: existing users keep their stored key (checkbox pre-checked) until they choose "Remove stored key". New behavior nudges session-only. Unresolved: whether the user wants forced migration (auto-purge stored keys) - left as his decision.

**Mock-mode unit tests (fallback evidence, not live-model proof).** tests/test_fallbacks.py runs against the REAL modules (agent/stock_sourcer.py, core/multimodal_embedder.py) with minimal dependency substitutes for absent packages (torch, cv2):
1. Keyless StockSourcer returns only source='fallback' candidates.
2. MultimodalEmbedder is_mock=True and embed_text returns a normalized 512-d vector.
3. Mock embeddings are deterministic across instantiations.
All 3 pass. Known limitation: torch is a declared hard dependency (requirements.txt); without it the app does not start, so the mock path covers missing model weights/models, not a missing torch install. Making torch optional was deliberately not attempted - it risks the real GPU path for no production benefit.

**Pricing grounding for the live smoke-test proposal (Oct 30, 2026).**
- Gemini image generation (app uses gemini-3.1-flash-image): official pricing page lists Free Tier "Not available"; paid $0.0168 per 1K image ($0.0252 2K / $0.0567 4K). Source: https://ai.google.dev/gemini-api/docs/pricing
- Gemini text/flash free tier exists for "certain models" with limited access; no claim made here that transcription calls are free.
- Pexels API: completely free, 200 requests/hour, 20,000/month default. Source: https://help.pexels.com/hc/en-us/articles/47677890260761-Is-the-Pexels-API-free-to-use

---

## v15 — Model list verified, upload limit harmonized, provider-error test (2026-10-30)

**Upload limit.** The Create-page "200MB per file" was a harness artifact (server launched from the wrong CWD, so .streamlit/config.toml never loaded). With the correct working directory the uploader shows "10GB per file", matching config (maxUploadSize=10000) and the pipeline page copy. Verified in-browser.

**Model list verified against ai.google.dev (Oct 30, 2026).** gemini-3.5-flash is officially DEPRECATED (auto-routes to gemini-3.6-flash; no longer selectable in AI Studio). UI now offers gemini-3.8-flash (latest), gemini-3.6-flash, gemini-3.5-flash-lite; transcribe options: gemini-3.5-transcribe, 3.8-flash, 3.6-flash. DEPRECATED_MODEL_MAP remaps stored "gemini-3.5-flash" to its official replacement instead of silently jumping to the newest model. Stale default "gemini-2.5-flash" replaced with gemini-3.8-flash. The test harness stub had its own outdated copy of these dicts - now mirrors the real module. Sources: https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash (deprecation notice), https://ai.google.dev/gemini-api/docs/models (endpoint table), https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-flash-image (gemini-3.1-flash-image id), https://ai.google.dev/gemini-api/docs/models/gemini-3.5-transcribe.

**Provider-error path tested through the real UI.** With a dependency substitute making every image call raise a realistic "404 NOT_FOUND: models/... is not found", the Create flow shows one red alert per failed image with the provider's message, finishes with "Created 0 of 3 images", and stays responsive (v16-model-error.png). No hang, no silent success.

**Smoke-test cost bound (grounded).** Sample: repo's sample_voiceover.wav, 10.0 s, 320 KB. Transcription: app default gemini-3.8-flash - official pricing lists Free Tier "Free of charge" for input and output; even at paid rates ($0.75/1M in, $3.75/1M out through Dec 31 2026) a 10 s clip (~320 audio tokens in, a few hundred out) is well under $0.01. Images: gemini-3.1-flash-image at default 1K, $0.0168/image (Free Tier "Not available" for image generation); 2 images = $0.0336. Worst-case total < $0.05; proposed cap $0.25 (5x headroom). Bounded by construction: fixed 10 s file, exactly one transcription call, image count hard-limited to 2, abort on first failure, no retries. Sources: https://ai.google.dev/gemini-api/docs/pricing and mirror https://aistudio.google.com/docs/pricing; Pexels free (200 req/hr, 20k/month): https://help.pexels.com/hc/en-us/articles/47677890260761-Is-the-Pexels-API-free-to-use

**Cap enforcement (v15 follow-up).** generate_image() and GeminiBackend.transcribe_audio() now accept an optional max_output_tokens (default None = production behavior unchanged). The proposed smoke test passes explicit caps: transcription 2,048 output tokens, images 2,560 each (a 1K image is ~1,120 tokens, leaving text headroom). Neither module contains any retry logic; the test is exactly 3 requests (1 transcription + 2 images), sequential, fixed inputs. Worst case at paid rates: transcription input <= ~570 tokens (979-char prompt template + ~320 audio tokens) + 2,048 out; images ~70 input tokens each + 2,560 out each - total < $0.09 at current promo rates, so $0.25 is a real >2.5x margin over the worst case, ~7x over expected (~$0.034). Default image resolution is 1K per https://ai.google.dev/gemini-api/docs/image-generation and the model page https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-image ("0.5K, 2K and 4K, default 1K"); the app sets no image_size.

**v15b.** Status/expanders icon font guard extended to Streamlit's stExpanderIcon* testids (status summaries rendered the literal ligature text "check" overlapping the label). tests/smoke_live.py added: the reproducible bounded runner for the proposed live check - exactly 3 requests max, aborts on first failure, refuses a 4th request, passes the explicit max_output_tokens caps. Walkthrough headline corrected to "Local workflow tested end to end; live AI check still pending" with a STATUS pill above the fold.

**v15c.** Create-flow status widget now matches the outcome: any failed image sets state="error" with "(N failed)" in the label (previously always showed a success check even at 0 of 3); when everything failed the box stays expanded so the per-image errors are visible. Walkthrough sources box now lists core/image_generator.py and tests/smoke_live.py.
