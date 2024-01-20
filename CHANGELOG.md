# Changelog

## [3.0.0](https://github.com/jlopatec/phoenix/compare/v2.5.0...v3.0.0) (2024-01-20)


### ⚠ BREAKING CHANGES

* Update `llm_classify` and `llm_generate` interfaces ([#1974](https://github.com/jlopatec/phoenix/issues/1974))
* **models:** openAI 1.0 ([#1716](https://github.com/jlopatec/phoenix/issues/1716))

### Features

* add `LLM_OUTPUT_MESSAGES` and rename `LLM_MESSAGES` to `LLM_INPUT_MESSAGES` ([#1462](https://github.com/jlopatec/phoenix/issues/1462)) ([f90ecb7](https://github.com/jlopatec/phoenix/commit/f90ecb7b094906b08f23e50053351d3bb54ebca6))
* Add a general purpose `LlamaIndex` debug callback handler ([#1608](https://github.com/jlopatec/phoenix/issues/1608)) ([f4697ac](https://github.com/jlopatec/phoenix/commit/f4697ac1010fb1345bbc5752bb985afecb0ed257))
* add ability to save and load TraceDatasets ([#2082](https://github.com/jlopatec/phoenix/issues/2082)) ([60c5e5e](https://github.com/jlopatec/phoenix/commit/60c5e5e1012680d62b1d439aaa822dd067578474))
* Add async submission to `llm_generate` ([#1965](https://github.com/jlopatec/phoenix/issues/1965)) ([5999133](https://github.com/jlopatec/phoenix/commit/59991331c93d4f994f00745124b73602d84cc95d))
* add computed values (e.g. latency) to filterable quantities ([#1609](https://github.com/jlopatec/phoenix/issues/1609)) ([53335c2](https://github.com/jlopatec/phoenix/commit/53335c2ec67c083a42cb4271393d0974fdec8dc6))
* Add default UMAP parameters in launch_app() ([#1224](https://github.com/jlopatec/phoenix/issues/1224)) ([dce7551](https://github.com/jlopatec/phoenix/commit/dce7551d00dd37e0ee005bafeed1757b0f5013bd))
* Add dockerfile ([#1761](https://github.com/jlopatec/phoenix/issues/1761)) ([4fa8929](https://github.com/jlopatec/phoenix/commit/4fa8929f4103e9961a8df0eb059b8df149ed648f))
* add get_trace_dataset method to session ([#2107](https://github.com/jlopatec/phoenix/issues/2107)) ([9754b60](https://github.com/jlopatec/phoenix/commit/9754b604af7ee8914333f6a59cabbeaf79f9eadf))
* add latency P50 and P99 to tracing page header ([#1519](https://github.com/jlopatec/phoenix/issues/1519)) ([767cd2a](https://github.com/jlopatec/phoenix/commit/767cd2a293ad8846d32489722cbce66f36478d9f))
* add long-context evaluators, including map reduce and refine patterns ([#1710](https://github.com/jlopatec/phoenix/issues/1710)) ([0c3b105](https://github.com/jlopatec/phoenix/commit/0c3b1053f95b88e234ed3ccfffa50b27f48dc359))
* Add OpenAI Rate limiting ([#1805](https://github.com/jlopatec/phoenix/issues/1805)) ([115e044](https://github.com/jlopatec/phoenix/commit/115e04478f7192bdb4aa7b7a1cd0a5bd950fb03c))
* add persistence for span evaluations ([#2021](https://github.com/jlopatec/phoenix/issues/2021)) ([589d482](https://github.com/jlopatec/phoenix/commit/589d482419b1119edfae24f837fd17d57612835f))
* Add processing functions file and token methods to OpenAIModel ([#1511](https://github.com/jlopatec/phoenix/issues/1511)) ([7063299](https://github.com/jlopatec/phoenix/commit/7063299e88880669f723a1192f16dd214b5ed426))
* Add retries to Bedrock ([#1927](https://github.com/jlopatec/phoenix/issues/1927)) ([2728c3e](https://github.com/jlopatec/phoenix/commit/2728c3e75927ca34e05c83336b3a8e9f5476466e))
* add support for explanations to run_evals ([#1975](https://github.com/jlopatec/phoenix/issues/1975)) ([5143529](https://github.com/jlopatec/phoenix/commit/51435297538c83542f7df6ba5860a50ef71307cf))
* Add support for Google's Gemini models via Vertex python sdk ([#2008](https://github.com/jlopatec/phoenix/issues/2008)) ([caf826c](https://github.com/jlopatec/phoenix/commit/caf826c8ced9f8840d3150f7d466bdc2295d2054))
* add templating semantic conventions to llm spans ([#1475](https://github.com/jlopatec/phoenix/issues/1475)) ([9ad15b3](https://github.com/jlopatec/phoenix/commit/9ad15b363430e7ba9c12919540eff9fe7eed79e5))
* Add VertexAI model implementation ([#1470](https://github.com/jlopatec/phoenix/issues/1470)) ([e688fbf](https://github.com/jlopatec/phoenix/commit/e688fbf1e7748bac49db7bcdda0f202b87fbfbf1))
* **app:** databricks notebook support ([#2086](https://github.com/jlopatec/phoenix/issues/2086)) ([b517480](https://github.com/jlopatec/phoenix/commit/b517480e18d5e5d4c10d237cf13aff7546aa6529))
* at exporter initialization, warn user if Phoenix is not running ([#1535](https://github.com/jlopatec/phoenix/issues/1535)) ([80b2385](https://github.com/jlopatec/phoenix/commit/80b23852b26fdf2d269cfd26a52b1677895b4ae2))
* **embeddings:** audio support ([#1920](https://github.com/jlopatec/phoenix/issues/1920)) ([61cc550](https://github.com/jlopatec/phoenix/commit/61cc55074c7381746886131c19e06d92a33f8489))
* **emb:** video URLs ([#1630](https://github.com/jlopatec/phoenix/issues/1630)) ([2c68368](https://github.com/jlopatec/phoenix/commit/2c6836825025f5f0154d3af367fd460167121488))
* enhance input types of run_relevance_eval ([#1504](https://github.com/jlopatec/phoenix/issues/1504)) ([b5be66e](https://github.com/jlopatec/phoenix/commit/b5be66e571b65699dd8f57042473c729b11afcab))
* error handling attributes for llama-index ([#1449](https://github.com/jlopatec/phoenix/issues/1449)) ([080a65d](https://github.com/jlopatec/phoenix/commit/080a65d48fdeb23e3c464ef8f5cc375e508d9c41))
* **eval:** add gpt-3.5-turbo-instruct to OpenAIModel ([#1613](https://github.com/jlopatec/phoenix/issues/1613)) ([ed89f5c](https://github.com/jlopatec/phoenix/commit/ed89f5c7ab802a925794215fdaab2085e5124527))
* Evals with explanations ([#1699](https://github.com/jlopatec/phoenix/issues/1699)) ([2db8141](https://github.com/jlopatec/phoenix/commit/2db814102ea27f441e201740cc75ace79c82837c))
* **evals:** add an output_parser to llm_generate ([#1736](https://github.com/jlopatec/phoenix/issues/1736)) ([6408dda](https://github.com/jlopatec/phoenix/commit/6408dda7d33bfe84d929ddaacd01cb3269e0f63a))
* **evals:** add Azure parameters to `OpenAIModel` ([#1566](https://github.com/jlopatec/phoenix/issues/1566)) ([469be0d](https://github.com/jlopatec/phoenix/commit/469be0d9dcf4b78d8a4ef912963138a264d4753d))
* **evals:** Add Bedrock model support to Evals ([#1567](https://github.com/jlopatec/phoenix/issues/1567)) ([35c4fe5](https://github.com/jlopatec/phoenix/commit/35c4fe5aa0bd973f53dc19abf823c1387f917dc8))
* **evals:** Human vs AI Evals ([#1850](https://github.com/jlopatec/phoenix/issues/1850)) ([e96bd27](https://github.com/jlopatec/phoenix/commit/e96bd27ed626a23187a92ddb34720a07ee689ad1))
* **evals:** in `llm_classify`, use function calling to constrain LLM outputs when available; `llm_classify` now returns a dataframe ([#1651](https://github.com/jlopatec/phoenix/issues/1651)) ([b796cb4](https://github.com/jlopatec/phoenix/commit/b796cb4057846574d0cd7a3e3c078b7f6ef5ee91))
* **evals:** return partial results when llm function is interrupted ([#1755](https://github.com/jlopatec/phoenix/issues/1755)) ([1fb0849](https://github.com/jlopatec/phoenix/commit/1fb0849a4e5f39c6afc90a1417300747a0bf4bf6))
* **evals:** show span evaluations in trace details slideout ([#1810](https://github.com/jlopatec/phoenix/issues/1810)) ([4f0e4dc](https://github.com/jlopatec/phoenix/commit/4f0e4dce35b779f2167581f281a0c70d61597f1d))
* evaluation column selectors ([#1932](https://github.com/jlopatec/phoenix/issues/1932)) ([ed07809](https://github.com/jlopatec/phoenix/commit/ed07809910e8b97387ef0729d843677e7e8c68a7))
* evaluation ingestion (no user-facing feature is added) ([#1764](https://github.com/jlopatec/phoenix/issues/1764)) ([7c4039b](https://github.com/jlopatec/phoenix/commit/7c4039b3d9a04a73b312a09ceeb95a73de9610ef))
* evaluator enhancements ([#2045](https://github.com/jlopatec/phoenix/issues/2045)) ([1cc9c0a](https://github.com/jlopatec/phoenix/commit/1cc9c0a7a32c5896ff8a83fe25defaeb01fd9bf2))
* feature flags context ([#1802](https://github.com/jlopatec/phoenix/issues/1802)) ([a2732cd](https://github.com/jlopatec/phoenix/commit/a2732cd115dad011c3856819fd2abf2a80ca2154))
* Implement asynchronous submission for OpenAI evals ([#1754](https://github.com/jlopatec/phoenix/issues/1754)) ([30c011d](https://github.com/jlopatec/phoenix/commit/30c011de471b68bc1a912780eff03ae0567d803e))
* **infra:** server-side template for index.html ([#1459](https://github.com/jlopatec/phoenix/issues/1459)) ([919bf20](https://github.com/jlopatec/phoenix/commit/919bf202ea1b41a104a4b2f8d8ad364d03e1c0b2))
* instantiate evaluators by criteria ([#1983](https://github.com/jlopatec/phoenix/issues/1983)) ([9c72616](https://github.com/jlopatec/phoenix/commit/9c72616aee88116c0194937d136bf3a74817132b))
* Instrument LlamaIndex streaming responses ([#1901](https://github.com/jlopatec/phoenix/issues/1901)) ([f46396e](https://github.com/jlopatec/phoenix/commit/f46396e04976475220092249a4e83f252d319630))
* launch phoenix with evaluations ([#2095](https://github.com/jlopatec/phoenix/issues/2095)) ([9656d0c](https://github.com/jlopatec/phoenix/commit/9656d0cc41bb995f2dfa35477ccf6569104da0c8))
* LiteLLM model support for evals ([#1675](https://github.com/jlopatec/phoenix/issues/1675)) ([5f2a999](https://github.com/jlopatec/phoenix/commit/5f2a9991059e060423853567a20789eba832f65a))
* **models:** openAI 1.0 ([#1716](https://github.com/jlopatec/phoenix/issues/1716)) ([2564521](https://github.com/jlopatec/phoenix/commit/2564521e1ce00aaabaf592ce3735a656e1c6f1b8))
* openai async streaming instrumentation ([#1900](https://github.com/jlopatec/phoenix/issues/1900)) ([06d643b](https://github.com/jlopatec/phoenix/commit/06d643b7c7255b79c7a7e4ea587b4e445122ac37))
* openai instrumentor with chat completions support ([#1550](https://github.com/jlopatec/phoenix/issues/1550)) ([26d2cd4](https://github.com/jlopatec/phoenix/commit/26d2cd49cda161c8cde7f0a9040efbf2835d91c0))
* openai streaming function call message support ([#1914](https://github.com/jlopatec/phoenix/issues/1914)) ([25279ca](https://github.com/jlopatec/phoenix/commit/25279ca563a81e438b7bbc3fd897d13ecca67b60))
* openai streaming spans show up in the ui ([#1888](https://github.com/jlopatec/phoenix/issues/1888)) ([ffa1d41](https://github.com/jlopatec/phoenix/commit/ffa1d41e633b6fee4978a9b705fa10bf4b5fe137))
* openai streaming tool calls ([#1936](https://github.com/jlopatec/phoenix/issues/1936)) ([6dd14cf](https://github.com/jlopatec/phoenix/commit/6dd14cf21acb12b599107619d0b7f4b0cdc55f4c))
* propagate error status codes to parent spans for improved visibility into trace exceptions ([#1824](https://github.com/jlopatec/phoenix/issues/1824)) ([1a234e9](https://github.com/jlopatec/phoenix/commit/1a234e902d5882f19ab3c497e788bb2c4e2ff227))
* Protect Langchain callbacks ([#1653](https://github.com/jlopatec/phoenix/issues/1653)) ([196c22b](https://github.com/jlopatec/phoenix/commit/196c22ba3a5b049b99312ece7b37c95ac1e4375e))
* **rag:** eval relevancy conventions ([#1612](https://github.com/jlopatec/phoenix/issues/1612)) ([e9f92f7](https://github.com/jlopatec/phoenix/commit/e9f92f7532ef05a885d66629b2d6e077ad82f39a))
* reference link correctness evaluation prompt template ([#1771](https://github.com/jlopatec/phoenix/issues/1771)) ([bf731df](https://github.com/jlopatec/phoenix/commit/bf731df32d0f908b91a9f3ffb5ed6dcf6e00ff13))
* sagemaker nobebook support ([#1772](https://github.com/jlopatec/phoenix/issues/1772)) ([2c0ffbc](https://github.com/jlopatec/phoenix/commit/2c0ffbc1479ae0255b72bc2d31d5f3204fd8e32c))
* semantic conventions for `tool_calls` array in OpenAI ChatCompletion messages ([#1837](https://github.com/jlopatec/phoenix/issues/1837)) ([c079f00](https://github.com/jlopatec/phoenix/commit/c079f00fd731e281671bace9ef4b68d4cbdcc584))
* support asynchronous chat completions for openai instrumentation ([#1849](https://github.com/jlopatec/phoenix/issues/1849)) ([f066e10](https://github.com/jlopatec/phoenix/commit/f066e108c07c95502ba77b11fcb37fe3e5e5ed72))
* support eval exports for session ([#2094](https://github.com/jlopatec/phoenix/issues/2094)) ([8757fa8](https://github.com/jlopatec/phoenix/commit/8757fa899428e13991acedb4beb77aadfaf42ea0))
* Support first-party Anthropic python SDK ([#2004](https://github.com/jlopatec/phoenix/issues/2004)) ([a323283](https://github.com/jlopatec/phoenix/commit/a323283021372bd3b7c95b435a8bef347e12028f))
* support function calling for run_evals ([#1978](https://github.com/jlopatec/phoenix/issues/1978)) ([8be325c](https://github.com/jlopatec/phoenix/commit/8be325cef3d48ce98f8c27c29dc46a42516ab970))
* support instrumentation for openai synchronous streaming ([#1879](https://github.com/jlopatec/phoenix/issues/1879)) ([b6e8c73](https://github.com/jlopatec/phoenix/commit/b6e8c732926ea112775e9541173a9bdb29482d8d))
* support multiclass classification ([#1629](https://github.com/jlopatec/phoenix/issues/1629)) ([7d9ae34](https://github.com/jlopatec/phoenix/commit/7d9ae347deb064480e7dcf259c7e47c0daa1bbc5))
* support running multiple evals at once ([#1742](https://github.com/jlopatec/phoenix/issues/1742)) ([79d4473](https://github.com/jlopatec/phoenix/commit/79d44739ba75b980967440a91850c9bed01ceb99))
* **trace:** add live updating UI, stream on/off switch ([#1554](https://github.com/jlopatec/phoenix/issues/1554)) ([92e53bf](https://github.com/jlopatec/phoenix/commit/92e53bfed54b7ccdbfae3cd0b84edba623abd612))
* **trace:** prompt template UI ([#1515](https://github.com/jlopatec/phoenix/issues/1515)) ([b7b1668](https://github.com/jlopatec/phoenix/commit/b7b166845d39a73d2b36c7f71035445171249d74))
* **traces:** add `v1/traces` HTTP endpoint to handle `ExportTraceServiceRequest` ([3c94dea](https://github.com/jlopatec/phoenix/commit/3c94dea6f5986c7b1658462d4700d21df56f72f6))
* **traces:** add `v1/traces` HTTP endpoint to handle `ExportTraceServiceRequest` ([#1968](https://github.com/jlopatec/phoenix/issues/1968)) ([3c94dea](https://github.com/jlopatec/phoenix/commit/3c94dea6f5986c7b1658462d4700d21df56f72f6))
* **traces:** add reranking span kind for document reranking in llama index ([#1588](https://github.com/jlopatec/phoenix/issues/1588)) ([1708748](https://github.com/jlopatec/phoenix/commit/1708748f81a6bc3c8015410e412b963bd9c1dad9))
* **traces:** add retrieval summary to header ([#2006](https://github.com/jlopatec/phoenix/issues/2006)) ([8af0582](https://github.com/jlopatec/phoenix/commit/8af0582c8c5cb19fe6bd84408f3d8ef26480c19e))
* **traces:** configurable endpoint for the exporter ([#1795](https://github.com/jlopatec/phoenix/issues/1795)) ([8515763](https://github.com/jlopatec/phoenix/commit/851576385f548d8515e745bb39392fcf1b070e93))
* **traces:** display document evaluations alongside the document ([#1823](https://github.com/jlopatec/phoenix/issues/1823)) ([2ca3613](https://github.com/jlopatec/phoenix/commit/2ca361348ad112c6320c17b40d36bfadb0c2c66f))
* **traces:** display document retrieval metrics on trace details ([#1902](https://github.com/jlopatec/phoenix/issues/1902)) ([0c35229](https://github.com/jlopatec/phoenix/commit/0c352297b3cef838651e69a05ae5357cbdbd61a5))
* **traces:** document retrieval metrics based on document evaluation scores ([#1826](https://github.com/jlopatec/phoenix/issues/1826)) ([3dfb7bd](https://github.com/jlopatec/phoenix/commit/3dfb7bdfba3eb61e57dba503efcc761511479a90))
* **traces:** document retrieval metrics on trace / span tables ([#1873](https://github.com/jlopatec/phoenix/issues/1873)) ([733d233](https://github.com/jlopatec/phoenix/commit/733d2339ec3ffabc9ac83454fb6540adfefe1526))
* **traces:** evaluation annotations on traces for associating spans with eval metrics ([#1693](https://github.com/jlopatec/phoenix/issues/1693)) ([a218a65](https://github.com/jlopatec/phoenix/commit/a218a650cefa8925ed7b6627c121454bfc94ec0d))
* **traces:** evaluation summary on the header ([#2000](https://github.com/jlopatec/phoenix/issues/2000)) ([965beb0](https://github.com/jlopatec/phoenix/commit/965beb00c223b405cd8c56c6a58d15beb391af78))
* **traces:** filterable span and document evaluation summaries ([#1880](https://github.com/jlopatec/phoenix/issues/1880)) ([f90919c](https://github.com/jlopatec/phoenix/commit/f90919c6162ce6bba3b12c5bba92b31f31128739))
* **traces:** graphql query for document evaluation summary ([#1874](https://github.com/jlopatec/phoenix/issues/1874)) ([8a6a063](https://github.com/jlopatec/phoenix/commit/8a6a06326e42f58018e030e0d854847d6fe6f10b))
* **trace:** show document metadata ([#1500](https://github.com/jlopatec/phoenix/issues/1500)) ([8ccf37d](https://github.com/jlopatec/phoenix/commit/8ccf37d98109c03fb339dad37bbac0435bcc4b6f))
* **trace:** show the model name in the LLM span info ([#1512](https://github.com/jlopatec/phoenix/issues/1512)) ([c6b118c](https://github.com/jlopatec/phoenix/commit/c6b118c1358cea899c7c7f30957b471b7c5c08b9))
* **traces:** query spans into dataframes ([#1910](https://github.com/jlopatec/phoenix/issues/1910)) ([6b51435](https://github.com/jlopatec/phoenix/commit/6b5143535cf4ad3d0149fb68234043d47debaa15))
* **traces:** server-side sort of spans by evaluation result (score or label) ([#1812](https://github.com/jlopatec/phoenix/issues/1812)) ([d139693](https://github.com/jlopatec/phoenix/commit/d1396931ab5b7b59c2777bb607afcf053134f6b7))
* **traces:** server-side span filter by evaluation result values ([#1858](https://github.com/jlopatec/phoenix/issues/1858)) ([6b05f96](https://github.com/jlopatec/phoenix/commit/6b05f96fa7fc328414daf3caaed7d807e018763a))
* **traces:** show all evaluations in the table" ([#1819](https://github.com/jlopatec/phoenix/issues/1819)) ([2b27333](https://github.com/jlopatec/phoenix/commit/2b273336b3448bc8cab1f433e79fc9fd868ad073))
* **traces:** span evaluation summary (aggregation metrics of scores and labels) ([#1846](https://github.com/jlopatec/phoenix/issues/1846)) ([5c5c3d6](https://github.com/jlopatec/phoenix/commit/5c5c3d69021fa21ce73a0d297107d5bf14fe4c98))
* **traces:** span table column visibility controls ([#1687](https://github.com/jlopatec/phoenix/issues/1687)) ([559852f](https://github.com/jlopatec/phoenix/commit/559852f12d976df691c24f3e89bf23a7650d148c))
* **traces:** Trace page header with latency, status, and evaluations ([#1831](https://github.com/jlopatec/phoenix/issues/1831)) ([1d88efd](https://github.com/jlopatec/phoenix/commit/1d88efdb623f5239106fde098fe51e53358592e2))
* **tracing:** add  LangChainInstrumentor for nested chains ([#1472](https://github.com/jlopatec/phoenix/issues/1472)) ([e5bf60e](https://github.com/jlopatec/phoenix/commit/e5bf60e385b381a2b45cb87aa3d60026fbe45163))
* **tracing:** gql validate_span_filter_condition ([#1578](https://github.com/jlopatec/phoenix/issues/1578)) ([a793c58](https://github.com/jlopatec/phoenix/commit/a793c5876bf0eef42c843b4153cda556a2dea243))
* **tracing:** trace and span filter / search field ([#1577](https://github.com/jlopatec/phoenix/issues/1577)) ([ca0a449](https://github.com/jlopatec/phoenix/commit/ca0a449c8cbc8d457f86bb0e9b94caca50eb1537))
* **ui:** add filter condition snippets ([#2049](https://github.com/jlopatec/phoenix/issues/2049)) ([567fa54](https://github.com/jlopatec/phoenix/commit/567fa5499a4584e03bf1695caea3179778d27889))
* **ui:** light theme ([#1657](https://github.com/jlopatec/phoenix/issues/1657)) ([db2aa46](https://github.com/jlopatec/phoenix/commit/db2aa4607689dab75ef3acf70fb8a24548c1c4a5))
* Update `llm_classify` and `llm_generate` interfaces ([#1974](https://github.com/jlopatec/phoenix/issues/1974)) ([9fd35a1](https://github.com/jlopatec/phoenix/commit/9fd35a11399c2d1aba1be40592cac0e31fff2d80))
* Verbose evals ([#1558](https://github.com/jlopatec/phoenix/issues/1558)) ([50e765b](https://github.com/jlopatec/phoenix/commit/50e765bb98dc84c540401dd14135104ec6582678))


### Bug Fixes

* `LlamaIndex` callback handler should fail gracefully ([#1606](https://github.com/jlopatec/phoenix/issues/1606)) ([a183cdc](https://github.com/jlopatec/phoenix/commit/a183cdc4588e99c35230d10f6204f8911365fbc8))
* add bedrock import ([#1695](https://github.com/jlopatec/phoenix/issues/1695)) ([dc7f3ef](https://github.com/jlopatec/phoenix/commit/dc7f3ef6fa7c184e46ef92f1c035a29345e0b12e))
* Add lock failsafe ([#1956](https://github.com/jlopatec/phoenix/issues/1956)) ([9ddbd9c](https://github.com/jlopatec/phoenix/commit/9ddbd9caa4a66ae47c5447aa4dafa46cc9c5cfa5))
* add name= to Run() for langchain tracer ([#1671](https://github.com/jlopatec/phoenix/issues/1671)) ([70c3b32](https://github.com/jlopatec/phoenix/commit/70c3b3241b29df573318bde823a9e37f4779ac10))
* add tiktoken to langchain_tracing_tutorial.ipynb ([#1463](https://github.com/jlopatec/phoenix/issues/1463)) ([a907233](https://github.com/jlopatec/phoenix/commit/a9072333cae1090fb11f4bbebc6c7945157283d2))
* Add uninmplemented methods to VertexAIModel from abstract class ([#1531](https://github.com/jlopatec/phoenix/issues/1531)) ([5b80d91](https://github.com/jlopatec/phoenix/commit/5b80d91a2dedd58f4ab0f556b26b834a39a8ff09))
* Adjust evaluation templates and rails for Gemini compatibility ([#2075](https://github.com/jlopatec/phoenix/issues/2075)) ([3a7bfd2](https://github.com/jlopatec/phoenix/commit/3a7bfd2158b5c0ea995f4d1ec00a6f7b2c4a21e8))
* allow streaming response to be iterated by user ([#1862](https://github.com/jlopatec/phoenix/issues/1862)) ([76a2443](https://github.com/jlopatec/phoenix/commit/76a24436d7f2d1cb56ac77fb8486b4296f65a615))
* case-insensitive rails parsing ([#1496](https://github.com/jlopatec/phoenix/issues/1496)) ([9765085](https://github.com/jlopatec/phoenix/commit/9765085799bb1f2ce463503c708f391e44dfd4a0))
* change llama-index repo url ([#1538](https://github.com/jlopatec/phoenix/issues/1538)) ([911d731](https://github.com/jlopatec/phoenix/commit/911d731c01f28599b65ea87c3eaba391469b57f5))
* Clean up vertex clients after event loop closure ([#2102](https://github.com/jlopatec/phoenix/issues/2102)) ([202c7ea](https://github.com/jlopatec/phoenix/commit/202c7eadf79acd40505885d891a02eb9331ff1e4))
* Determine default async concurrency on a per-model basis ([#2096](https://github.com/jlopatec/phoenix/issues/2096)) ([b44d8aa](https://github.com/jlopatec/phoenix/commit/b44d8aa8de6854632f241000be244c43a26ed85e))
* Do not retry if eval was successful when using SyncExecutor ([#2016](https://github.com/jlopatec/phoenix/issues/2016)) ([a869190](https://github.com/jlopatec/phoenix/commit/a8691905e16fe1c8c6483d837399e8c499ce71cf))
* enhance llama-index callback support for exception events ([#1814](https://github.com/jlopatec/phoenix/issues/1814)) ([8db01df](https://github.com/jlopatec/phoenix/commit/8db01df096b5955adb12a57101beb76c943d8649))
* ensure float values are properly encoded by otel tracer ([#2024](https://github.com/jlopatec/phoenix/issues/2024)) ([b12a894](https://github.com/jlopatec/phoenix/commit/b12a89496468063e2cd6a99e1ea08d49af5c6ba1))
* ensure llamaindex spans are correctly encoded ([#2023](https://github.com/jlopatec/phoenix/issues/2023)) ([3ca6262](https://github.com/jlopatec/phoenix/commit/3ca6262ad38c13b63dc2cde84e1a97a76ba1c323))
* **evals:** Set client type in bedrock.py for constructor ([#1595](https://github.com/jlopatec/phoenix/issues/1595)) ([0a3aad5](https://github.com/jlopatec/phoenix/commit/0a3aad56e8607a5ba4da186685a9c98e34d06880))
* **evals:** update prompt template for summarization eval and rails mapping to use lower case "good" / "bad" ([#1562](https://github.com/jlopatec/phoenix/issues/1562)) ([8bad5ac](https://github.com/jlopatec/phoenix/commit/8bad5ac2ade211afda75d4a5801e1080479a2d40))
* failed imports from typing_extensions 4.5.0 in Colab ([#1620](https://github.com/jlopatec/phoenix/issues/1620)) ([f57c4bb](https://github.com/jlopatec/phoenix/commit/f57c4bb178eaf9c9ba95c2fc86e8cc56c73f8061))
* Handle missing vertex candidates ([#2055](https://github.com/jlopatec/phoenix/issues/2055)) ([1d0475a](https://github.com/jlopatec/phoenix/commit/1d0475afa4974eb9590c70334961996f99e6d856))
* increment version to v0.0.50rc5 ([#1649](https://github.com/jlopatec/phoenix/issues/1649)) ([664e207](https://github.com/jlopatec/phoenix/commit/664e207b60bed36b84b91fc7c811e034ce2a17d8))
* llama-index extra ([#1958](https://github.com/jlopatec/phoenix/issues/1958)) ([d9b68eb](https://github.com/jlopatec/phoenix/commit/d9b68eb2e8f3422a491dc8a0981dd474d84d6260))
* LlamaIndex callback handler drops events if the `start_event` handler is not called ([#1668](https://github.com/jlopatec/phoenix/issues/1668)) ([428053f](https://github.com/jlopatec/phoenix/commit/428053fcefdd9b5502b3fc38ac0235a0e938c8a1))
* LlamaIndex compatibility fix ([#1940](https://github.com/jlopatec/phoenix/issues/1940)) ([052349d](https://github.com/jlopatec/phoenix/commit/052349d594fa2d25b827f5318e0b438f33bf165c))
* make alert icon for exceptions visible ([#2001](https://github.com/jlopatec/phoenix/issues/2001)) ([e7a6567](https://github.com/jlopatec/phoenix/commit/e7a6567f813149e4e7952f4b80183f39ea7cc15f))
* make the app launchable when nest_asyncio is applied ([#1783](https://github.com/jlopatec/phoenix/issues/1783)) ([f9d5085](https://github.com/jlopatec/phoenix/commit/f9d508510c739007243ca200560268d53e6cb543))
* Model stability enhancements ([#1939](https://github.com/jlopatec/phoenix/issues/1939)) ([dca42e0](https://github.com/jlopatec/phoenix/commit/dca42e026acc079f0523158d10e2149b01136600))
* **notebook:** missing experimental in pip install ([#1549](https://github.com/jlopatec/phoenix/issues/1549)) ([477f049](https://github.com/jlopatec/phoenix/commit/477f0499635e7b11df80e7164a878d9f035bb48a))
* OpenAI clients are not cleaned up after calls to `llm_classify` ([#2068](https://github.com/jlopatec/phoenix/issues/2068)) ([3233d56](https://github.com/jlopatec/phoenix/commit/3233d561783556ef8799e65fe1335da52d161a12))
* pin llama-index temporarily ([#1806](https://github.com/jlopatec/phoenix/issues/1806)) ([d6aa76e](https://github.com/jlopatec/phoenix/commit/d6aa76e2707528c367ea9ec5accc503871f97644))
* pin openai version below 1.0.0 ([#1714](https://github.com/jlopatec/phoenix/issues/1714)) ([d21e364](https://github.com/jlopatec/phoenix/commit/d21e36459a42f01d04a373236fef2e603eea159d))
* precision at k calculation in notebooks ([#1592](https://github.com/jlopatec/phoenix/issues/1592)) ([c558363](https://github.com/jlopatec/phoenix/commit/c558363dfb191b9e4f83cc64eac3ff0d5b64b07e))
* Rate limiter will sometimes tank request rate with many concurrent requests ([#1855](https://github.com/jlopatec/phoenix/issues/1855)) ([2530569](https://github.com/jlopatec/phoenix/commit/25305699c639d4c556c413e27a4c13378a548a77))
* remove html as ensured-targets ([2d5bfda](https://github.com/jlopatec/phoenix/commit/2d5bfda14e6ecef2ffc53555fce9b0c5fd31bffe))
* Remove LiteLLM model support check ([#2046](https://github.com/jlopatec/phoenix/issues/2046)) ([45d3fe6](https://github.com/jlopatec/phoenix/commit/45d3fe6e1a2aaf931eb783aaca68525579215386))
* remove sklearn metrics not available in sagemaker ([#1791](https://github.com/jlopatec/phoenix/issues/1791)) ([20ab6e5](https://github.com/jlopatec/phoenix/commit/20ab6e551eb4de16df65d31d10b8efe34814c866))
* restore process session ([#1781](https://github.com/jlopatec/phoenix/issues/1781)) ([34a32c3](https://github.com/jlopatec/phoenix/commit/34a32c3e8567672bd1ac0979923566c39adecfcf))
* show localhost when the notebook is running locally ([#2090](https://github.com/jlopatec/phoenix/issues/2090)) ([095298d](https://github.com/jlopatec/phoenix/commit/095298d88bc44b28928b44b2b593eca9d54e843b))
* streaming initial ([#1624](https://github.com/jlopatec/phoenix/issues/1624)) ([6a508fd](https://github.com/jlopatec/phoenix/commit/6a508fd8dc9d97a1251f4a677a848ec9f5e55b3f))
* toxicity notebook bug ([#1505](https://github.com/jlopatec/phoenix/issues/1505)) ([324c585](https://github.com/jlopatec/phoenix/commit/324c585feea713aebe4231c286bff2a7a56f7baa))
* trace dataset to disc ([#1798](https://github.com/jlopatec/phoenix/issues/1798)) ([278d344](https://github.com/jlopatec/phoenix/commit/278d344434d43d5d05cc66abfcb9646b0ac2fb6d))
* **traces:** convert (non-list) iterables to lists during protobuf construction due to potential presence of ndarray when reading from parquet files ([#1801](https://github.com/jlopatec/phoenix/issues/1801)) ([ca72747](https://github.com/jlopatec/phoenix/commit/ca72747991bf60881d76f95e88c656b1cecff2df))
* **traces:** handle `AIMessageChunk` in langchain tracer by matching prefix in name ([#1724](https://github.com/jlopatec/phoenix/issues/1724)) ([8654c0a](https://github.com/jlopatec/phoenix/commit/8654c0a0c9e088284b4ed0bbbae4f571ae11b1e7))
* **traces:** Keep traces visible behind the details slideover ([#1709](https://github.com/jlopatec/phoenix/issues/1709)) ([1c8b8f1](https://github.com/jlopatec/phoenix/commit/1c8b8f175e0cc8506490a8e0e36ad92d7e7ff69a))
* **traces:** make column selector sync'd between tabs ([#1816](https://github.com/jlopatec/phoenix/issues/1816)) ([125431a](https://github.com/jlopatec/phoenix/commit/125431a15cb9eaf07db406a936bd38c42f8665d8))
* **traces:** prevent missing key exception when extracting invocation parameters in llama-index ([#2076](https://github.com/jlopatec/phoenix/issues/2076)) ([5cc9560](https://github.com/jlopatec/phoenix/commit/5cc956057f01613ebd08b0247d773c74b24a5aa3))
* **traces:** refetch traces and spans even when searching ([#1683](https://github.com/jlopatec/phoenix/issues/1683)) ([8d95b6d](https://github.com/jlopatec/phoenix/commit/8d95b6dc1cebdb88bbc657afdd95dbd3385105ae))
* **traces:** remove nan from log_evaluations ([#2056](https://github.com/jlopatec/phoenix/issues/2056)) ([df9ed5c](https://github.com/jlopatec/phoenix/commit/df9ed5cfd68da1e42d52b01aa9f85d68698c3b64))
* **traces:** span evaluations missing from the header ([#1908](https://github.com/jlopatec/phoenix/issues/1908)) ([5ace81e](https://github.com/jlopatec/phoenix/commit/5ace81e1a99afc72b6baf6464f1c21eea05eecdd))
* **traces:** span summary root span filter ([#1981](https://github.com/jlopatec/phoenix/issues/1981)) ([d286f07](https://github.com/jlopatec/phoenix/commit/d286f077e8493c98a795ae83066e8c1c43a5291f))
* **traces:** support token counts for custom models in llama_index ([#1673](https://github.com/jlopatec/phoenix/issues/1673)) ([1f5aeec](https://github.com/jlopatec/phoenix/commit/1f5aeecaa784bb3b714087df48707ca1f6db5b94))
* **tracing:** suspense boundary causing search to lose focus ([#1622](https://github.com/jlopatec/phoenix/issues/1622)) ([c744614](https://github.com/jlopatec/phoenix/commit/c744614e778af39c3be714c1745210b870cb0984))
* type check for on_event_start for llama-index ([#1557](https://github.com/jlopatec/phoenix/issues/1557)) ([fd709de](https://github.com/jlopatec/phoenix/commit/fd709defcc86843662318fb7878be2fc7edd3ea4))
* **UI:** fix icon and label colors for light mode ([#1678](https://github.com/jlopatec/phoenix/issues/1678)) ([27e9b81](https://github.com/jlopatec/phoenix/commit/27e9b8178b243ee28d4538df8ad50168d5e26134))
* unpin llama-index version in tutorial notebooks ([#1766](https://github.com/jlopatec/phoenix/issues/1766)) ([5ff74e3](https://github.com/jlopatec/phoenix/commit/5ff74e3895f1b0c5642bd0897dd65e6f2913a7bd))
* Update llama_index_openai_agent_tracing_tutorial.ipynb ([df6af4d](https://github.com/jlopatec/phoenix/commit/df6af4dd97fc41fb16971f5275924bcae87be6fc))
* update tracer for llama-index 0.9.0 ([#1750](https://github.com/jlopatec/phoenix/issues/1750)) ([48d0996](https://github.com/jlopatec/phoenix/commit/48d09960855d59419edfd10925aaa895fd370a0d))
* Use separate versioning file ([#2020](https://github.com/jlopatec/phoenix/issues/2020)) ([f38eedf](https://github.com/jlopatec/phoenix/commit/f38eedfbd57da6e9512f9aa9fcc5c97232b2bb6e))


### Documentation

* Add anyscale tutorial ([#1941](https://github.com/jlopatec/phoenix/issues/1941)) ([e47c8d0](https://github.com/jlopatec/phoenix/commit/e47c8d0be941f82ac13c5686fe75622cb59974ab))
* add back in topics column to evals notebook section (GITBOOK-276) ([47ea65c](https://github.com/jlopatec/phoenix/commit/47ea65c7ee7df0707d6b7b799e6a60f8702175f7))
* add code tour for tabular metrics ([#1532](https://github.com/jlopatec/phoenix/issues/1532)) ([d01cfe8](https://github.com/jlopatec/phoenix/commit/d01cfe8bfc6542b2d275d262070108fe0f90dac8))
* Add demo link, examples getting started (GITBOOK-396) ([e987315](https://github.com/jlopatec/phoenix/commit/e987315ab35550350921257aa3e38f5cb6796f3f))
* Add Eval models (GITBOOK-300) ([3bd455d](https://github.com/jlopatec/phoenix/commit/3bd455de02728bab1f4bba765bc0dcd578f984e0))
* add eval notebook links (GITBOOK-273) ([6cfdf22](https://github.com/jlopatec/phoenix/commit/6cfdf22fa42e68d1158b9d0e8d314569122f0022))
* Add Evaluating Traces Section (GITBOOK-386) ([7d72029](https://github.com/jlopatec/phoenix/commit/7d720293cc30e0021596c427081e3b6bef4b7f5b))
* Add evaluations section for results (GITBOOK-387) ([2e74be0](https://github.com/jlopatec/phoenix/commit/2e74be0a70a63ddfd6e42d84042f51fc9894f1cb))
* Add final thoughts to evaluation (GITBOOK-405) ([20eab16](https://github.com/jlopatec/phoenix/commit/20eab16c2dc99b3c6defd497120ee2472eb97ff1))
* add getpass ([#1523](https://github.com/jlopatec/phoenix/issues/1523)) ([d114533](https://github.com/jlopatec/phoenix/commit/d114533a50d998e137604ca900d95ddee5120465))
* add images to cards in integrations (GITBOOK-306) ([5e1dc05](https://github.com/jlopatec/phoenix/commit/5e1dc05a25dec14ca2c359809edef612cebec059))
* add import statement (GITBOOK-408) ([23247d7](https://github.com/jlopatec/phoenix/commit/23247d755bad33b618e683687dd297e95274939c))
* add instructions for docker build ([#1770](https://github.com/jlopatec/phoenix/issues/1770)) ([45eb5f2](https://github.com/jlopatec/phoenix/commit/45eb5f244997d0ff0e991879c297b564e46c9a18))
* add link (GITBOOK-403) ([0be280a](https://github.com/jlopatec/phoenix/commit/0be280a974e7b8e88904c35285b149cde2369804))
* Add LLM Tracing+Evals notebook with keyless example ([#1928](https://github.com/jlopatec/phoenix/issues/1928)) ([4c4aac6](https://github.com/jlopatec/phoenix/commit/4c4aac6425af851b68f52d537813a8a1293a2a4b))
* add notebook link for  (GITBOOK-329) ([4fcdeb8](https://github.com/jlopatec/phoenix/commit/4fcdeb81e8fd3e8dcf5f22e21de9ed5b950ee95d))
* Add prompt template page (GITBOOK-296) ([a7bbb8d](https://github.com/jlopatec/phoenix/commit/a7bbb8de38fded735be22d341dd84060904e7863))
* Add reranker to span kinds (GITBOOK-348) ([fee736d](https://github.com/jlopatec/phoenix/commit/fee736df2c212a4fe94a489b680693525b0446f8))
* add session info for notebooks ([#1490](https://github.com/jlopatec/phoenix/issues/1490)) ([06a5631](https://github.com/jlopatec/phoenix/commit/06a56318b78ca11e5df9970255f1dab1acd2db3a))
* Add Structured Extraction Use Case (GITBOOK-337) ([7a92ad2](https://github.com/jlopatec/phoenix/commit/7a92ad23f62fb5e056cb2867887e1e28c128e8aa))
* add tracing notebooks (GITBOOK-275) ([76c0e9f](https://github.com/jlopatec/phoenix/commit/76c0e9f70086e075c335566749cd7ad6bb16936e))
* add-relevance-notebook-links (GITBOOK-272) ([8eafd39](https://github.com/jlopatec/phoenix/commit/8eafd398e85a97e77f88783e81b28cb912f207f2))
* add-sections-to-notebook-page (GITBOOK-271) ([317279d](https://github.com/jlopatec/phoenix/commit/317279d8edd2a1e7b0257c7b26c645e9942e3913))
* autogen link ([#1946](https://github.com/jlopatec/phoenix/issues/1946)) ([c3fb4ce](https://github.com/jlopatec/phoenix/commit/c3fb4ce80005c41201824114400abcdf007574c0))
* bedrock model (GITBOOK-327) ([dbd6077](https://github.com/jlopatec/phoenix/commit/dbd60778129022c9f57f4e312b5d120b82869720))
* Benchmarking retrieval (GITBOOK-334) ([0da10f6](https://github.com/jlopatec/phoenix/commit/0da10f6119c4e064dc214b02b8ec140ea89ffcf8))
* Change order of links (GITBOOK-287) ([48e8ac1](https://github.com/jlopatec/phoenix/commit/48e8ac1f57026d25d2f60b276350ab98f228effb))
* Claud Hallucination (GITBOOK-354) ([480c907](https://github.com/jlopatec/phoenix/commit/480c9074fe63d7bbd02226824729fbfb74971842))
* Claude V2 Metrics & Bedrock (GITBOOK-353) ([cd02879](https://github.com/jlopatec/phoenix/commit/cd02879039355bad6029ed8fe08b79e5ca2d5d56))
* cleaned up ipynb notebooks to cleanly work in Colab environment ([#1530](https://github.com/jlopatec/phoenix/issues/1530)) ([c4927ac](https://github.com/jlopatec/phoenix/commit/c4927ac10a2984233714ff15a3e9ac1036586481))
* Clear anyscale tutorial outputs ([#1942](https://github.com/jlopatec/phoenix/issues/1942)) ([63580a6](https://github.com/jlopatec/phoenix/commit/63580a6bde204247953a920e8ca13209bd1d7f0b))
* contributing guide ([#1605](https://github.com/jlopatec/phoenix/issues/1605)) ([ac6774e](https://github.com/jlopatec/phoenix/commit/ac6774e79db8043273fcd342880120029d72a71e))
* default_umap_parameters (GITBOOK-325) ([266a1f4](https://github.com/jlopatec/phoenix/commit/266a1f45bdbe94ef38ebf8e467f2cc5cbe22a5e9))
* delete contributor reference section (GITBOOK-315) ([8c95feb](https://github.com/jlopatec/phoenix/commit/8c95febccf6c66daa94bfa7e4cd7cfe1660b1a21))
* describe Azure initialization for OpenAIModel (GITBOOK-324) ([9bfc9bc](https://github.com/jlopatec/phoenix/commit/9bfc9bcb92f43a2256bbf3fa9249e6e047d9b091))
* Environment documentation (GITBOOK-370) ([dbbb0a7](https://github.com/jlopatec/phoenix/commit/dbbb0a7cf86200d71327a2d29e889f31c1a0f149))
* eval concepts typo (GITBOOK-394) ([7c80d4b](https://github.com/jlopatec/phoenix/commit/7c80d4b3a151ee3a9b1937d04b7362ab2dba3697))
* eval concepts typos (GITBOOK-393) ([62bc99f](https://github.com/jlopatec/phoenix/commit/62bc99f9e4d729e3bd911f346ce58bf44141c30f))
* Eval Diffentiators (GITBOOK-376) ([bcd14a4](https://github.com/jlopatec/phoenix/commit/bcd14a4b0d0e3db5a716b19352e831ad8bc7da74))
* Eval results update (GITBOOK-363) ([6ec1c0e](https://github.com/jlopatec/phoenix/commit/6ec1c0e85d823771ae5f515e0fb39947156e8f9d))
* **evals:** document llm_generate with output parser ([#1741](https://github.com/jlopatec/phoenix/issues/1741)) ([1e70ec3](https://github.com/jlopatec/phoenix/commit/1e70ec3ff6158ffada40726419ae436ba6c7948d))
* **evals:** update RAG evaluations notebook ([#2092](https://github.com/jlopatec/phoenix/issues/2092)) ([9ad797a](https://github.com/jlopatec/phoenix/commit/9ad797ab8d2b726088c5bdb6e104e7bc1faf8509))
* **evals:** update ragas integration notebook ([#2100](https://github.com/jlopatec/phoenix/issues/2100)) ([66fb048](https://github.com/jlopatec/phoenix/commit/66fb0480a886bd92246d2bffedd32c66a33946ce))
* evaluation concepts typo fix (GITBOOK-390) ([2cbc1dc](https://github.com/jlopatec/phoenix/commit/2cbc1dcce6ac781d71fbf3107e6af74121be018e))
* Explanations (GITBOOK-371) ([5f33da3](https://github.com/jlopatec/phoenix/commit/5f33da313625e5231619a447a8fbe0d173d638b0))
* export data section (GITBOOK-286) ([eebb732](https://github.com/jlopatec/phoenix/commit/eebb732db24f13d9a77d538b24598ba6f593f632))
* Extract Data from Spans (GITBOOK-383) ([440f530](https://github.com/jlopatec/phoenix/commit/440f530e35c13c7be1f69a05b0819afec094c5ed))
* fix broken section link (GITBOOK-409) ([fee537b](https://github.com/jlopatec/phoenix/commit/fee537b9ede6a8d7899744b834a46066f7f8837d))
* Fix links on Phoenix Inferences (GITBOOK-359) ([0b2b401](https://github.com/jlopatec/phoenix/commit/0b2b401786ddb4a026f1f88fb13eefc171ac1228))
* Fix notebook output (GITBOOK-380) ([358adf8](https://github.com/jlopatec/phoenix/commit/358adf8a23e5823ac3230f83f41b3b71fcbecdac))
* fix order of arguments in llm_eval_binary (GITBOOK-322) ([91a608d](https://github.com/jlopatec/phoenix/commit/91a608d97a96bb641c248be82c19285a4a6ee022))
* fix slugs (GITBOOK-304) ([7f5e427](https://github.com/jlopatec/phoenix/commit/7f5e427b42fef50d880422469de99a9f46f7cf0f))
* fix typo (GITBOOK-314) ([1a007bb](https://github.com/jlopatec/phoenix/commit/1a007bb5f82c7b25a36b384c76a4c83217d137e7))
* fix typo (GITBOOK-318) ([99df8ad](https://github.com/jlopatec/phoenix/commit/99df8adc2e71cde503eba99a7f1a91a09e849d2a))
* fix typos (GITBOOK-317) ([080c8e7](https://github.com/jlopatec/phoenix/commit/080c8e74a87998de2d60bec02a2b821fb178f503))
* fix typos (GITBOOK-391) ([c8f5a55](https://github.com/jlopatec/phoenix/commit/c8f5a55d11669b3837c32cdbc10f99388f63276e))
* fix typos (GITBOOK-402) ([3cd973d](https://github.com/jlopatec/phoenix/commit/3cd973d8b0348bb580455f093cc36f58deecf1b9))
* fix typos (GITBOOK-406) ([eaa9bea](https://github.com/jlopatec/phoenix/commit/eaa9beac66cd72ad99ab5a8bc2aec95e2a4a0e7f))
* fix typos (GITBOOK-407) ([cad4820](https://github.com/jlopatec/phoenix/commit/cad4820c085c7799adcf8d72b4209440b37fcbb9))
* fix value prop (GITBOOK-293) ([7f5d834](https://github.com/jlopatec/phoenix/commit/7f5d8347a9e87e1b1990bdf9b4ca1899893491c0))
* fix wording on tracing notebook ([4b56e81](https://github.com/jlopatec/phoenix/commit/4b56e8196c44741cd1bfd1a934977c7cb5e160df))
* human vs AI (GITBOOK-375) ([d159b11](https://github.com/jlopatec/phoenix/commit/d159b11b3c4fddfb32374eb47320506fea42b579))
* Human vs AI edits (GITBOOK-377) ([686df61](https://github.com/jlopatec/phoenix/commit/686df614d1044b5edbac5cffa1ddf94fc503aea0))
* Improve rag evaluation (GITBOOK-379) ([e145b98](https://github.com/jlopatec/phoenix/commit/e145b980f08f1462ce45c4ddf8520aded795e124))
* Initial draft of evaluation core concept (GITBOOK-385) ([67369cf](https://github.com/jlopatec/phoenix/commit/67369cf34e5eaddf30e0e3093f163483643f14bb))
* Install section (GITBOOK-285) ([db5e580](https://github.com/jlopatec/phoenix/commit/db5e580640d2d3e793b74450e29996b275ce2a92))
* Landing Page + Quickstart (GITBOOK-282) ([a200451](https://github.com/jlopatec/phoenix/commit/a2004519dbf1a2b59527acc8e6067b2376803328))
* langchain agent tutorial ([#1474](https://github.com/jlopatec/phoenix/issues/1474)) ([fddfbf4](https://github.com/jlopatec/phoenix/commit/fddfbf4a562d1e87732053b6493a8d9778fb3caa))
* LangChainInstrumentor (GITBOOK-283) ([3afdd6b](https://github.com/jlopatec/phoenix/commit/3afdd6bd63a55a6735a7f459240b369be1d06ac4))
* LangChainInstrumentor (GITBOOK-297) ([ad11a76](https://github.com/jlopatec/phoenix/commit/ad11a765652441165fad619afe8f6f7d0e5dae44))
* LangChainInstrumentor in basics (GITBOOK-284) ([6f2750b](https://github.com/jlopatec/phoenix/commit/6f2750bc257bf155ff086b2068dec48501b45108))
* Llama Index OpenAI Agent tutorial edits ([#1466](https://github.com/jlopatec/phoenix/issues/1466)) ([1a13b71](https://github.com/jlopatec/phoenix/commit/1a13b71d4c2176daa20f05a8df974b83ebf6466b))
* LlamaIndex one-click (GITBOOK-298) ([4c43268](https://github.com/jlopatec/phoenix/commit/4c43268687acc730cf9b2f03fe21e322a8853657))
* LLM Evals - How to build an eval  (GITBOOK-335) ([4891c67](https://github.com/jlopatec/phoenix/commit/4891c6785a3f25cc8b8c5bd4c90ab8ba3d86a128))
* llm observability formatting (GITBOOK-288) ([d67e5bf](https://github.com/jlopatec/phoenix/commit/d67e5bf821e740e03e7186582a1302639686dcd6))
* LLM Observability update (GITBOOK-252) ([52755f6](https://github.com/jlopatec/phoenix/commit/52755f61c17673985be851aa1f11579a5650e45b))
* llm ops overview notebook ([#1882](https://github.com/jlopatec/phoenix/issues/1882)) ([5d15c3c](https://github.com/jlopatec/phoenix/commit/5d15c3c665583848624882e6d67979148673fca6))
* LLM Traces Section (GITBOOK-302) ([9ca4257](https://github.com/jlopatec/phoenix/commit/9ca4257c7ebb661431a32308d0cb4118cac03a41))
* llm traces wording ([b75fa7c](https://github.com/jlopatec/phoenix/commit/b75fa7c2221b1fdfbfa0c4dcb93add3819c4203b))
* Log Evaluations (GITBOOK-389) ([369d79d](https://github.com/jlopatec/phoenix/commit/369d79d479b4b094dbeeef8715b6127babb1a48c))
* Mistake Label of Precision (GITBOOK-364) ([75f8a27](https://github.com/jlopatec/phoenix/commit/75f8a2781249a395795b4c5497443615017c650b))
* Move autogen to integrations (GITBOOK-382) ([40d9426](https://github.com/jlopatec/phoenix/commit/40d9426e581ae92f0cac5a8bca443368cbc6ab4c))
* No subject (GITBOOK-270) ([137f842](https://github.com/jlopatec/phoenix/commit/137f842f85efdb6d7cb036a2008accb56952cbba))
* No subject (GITBOOK-280) ([d901dbb](https://github.com/jlopatec/phoenix/commit/d901dbbb25548dae86e099f437c381e9a16ed1b5))
* No subject (GITBOOK-294) ([8f96c41](https://github.com/jlopatec/phoenix/commit/8f96c41d898121e41ec7699a91bb88904c1b71c9))
* No subject (GITBOOK-323) ([550efc2](https://github.com/jlopatec/phoenix/commit/550efc237a1d1064525ca4b378dca661e4e531e7))
* No subject (GITBOOK-333) ([cf3b623](https://github.com/jlopatec/phoenix/commit/cf3b623b58688f462f80d53a920a81cade7967b9))
* No subject (GITBOOK-340) ([aea6b2e](https://github.com/jlopatec/phoenix/commit/aea6b2ed26848612840ab680e2fb3e04f40f6bcf))
* No subject (GITBOOK-341) ([d816fa1](https://github.com/jlopatec/phoenix/commit/d816fa18c5d93a0b8bc493eb724e77b400a332f1))
* No subject (GITBOOK-342) ([096efcc](https://github.com/jlopatec/phoenix/commit/096efcc7d062f8925c8b5dd66c01c2882d2e5aca))
* No subject (GITBOOK-347) ([b90efe8](https://github.com/jlopatec/phoenix/commit/b90efe8d30478dda1741e101336cb24203e1d12d))
* No subject (GITBOOK-350) ([34dec7a](https://github.com/jlopatec/phoenix/commit/34dec7aba101060e7d8c185c4534051c99e32068))
* No subject (GITBOOK-355) ([4c30859](https://github.com/jlopatec/phoenix/commit/4c30859df43fe2e51cecf7f63121c10e46f1bf6d))
* No subject (GITBOOK-361) ([440b61a](https://github.com/jlopatec/phoenix/commit/440b61a340a4ca6f8c0b84f4cf3d72915e786262))
* No subject (GITBOOK-365) ([d6f44da](https://github.com/jlopatec/phoenix/commit/d6f44daae84c4f98a45b30c97475df5edc2f3f36))
* No subject (GITBOOK-366) ([025d52b](https://github.com/jlopatec/phoenix/commit/025d52b7d9cf4a8c602ebacbf02a5e9c0cd471b6))
* No subject (GITBOOK-369) ([656b5c0](https://github.com/jlopatec/phoenix/commit/656b5c0b9164517f78488b15dec14517590b4d5e))
* No subject (GITBOOK-381) ([de90467](https://github.com/jlopatec/phoenix/commit/de90467a1a709296098832efeb79f610d709d940))
* No subject (GITBOOK-399) ([94df884](https://github.com/jlopatec/phoenix/commit/94df884aa6f7ee98610026d827d48b57c0ba032b))
* OpenAI Instrumentation (GITBOOK-328) ([754fc23](https://github.com/jlopatec/phoenix/commit/754fc2326ddd316eca4d67718c63adae1a21d2f4))
* pin tutorials to openai&lt;1 ([#1718](https://github.com/jlopatec/phoenix/issues/1718)) ([831c041](https://github.com/jlopatec/phoenix/commit/831c041185e5f514fa1a13836758efecbf05e1dd))
* Preserve output, clean eval notebooks ([#1590](https://github.com/jlopatec/phoenix/issues/1590)) ([c092387](https://github.com/jlopatec/phoenix/commit/c0923871304af0d73f12fa4474a2fe0bb2c946a9))
* RAG Evaluation (GITBOOK-378) ([429f537](https://github.com/jlopatec/phoenix/commit/429f537ee034639dc66c4c80b455b22c65a2e7bd))
* RAG evaluation notebook using traces ([#1857](https://github.com/jlopatec/phoenix/issues/1857)) ([4b67805](https://github.com/jlopatec/phoenix/commit/4b67805931b059635997326de596d46a0bad1b76))
* rag notebook links (GITBOOK-277) ([ab12cc6](https://github.com/jlopatec/phoenix/commit/ab12cc6ccc6bb7c1928c6b1476e36738b8ec99ba))
* **rag:** Add RAG eval script and notebook ([#1520](https://github.com/jlopatec/phoenix/issues/1520)) ([6ab192a](https://github.com/jlopatec/phoenix/commit/6ab192a63dda658ae0c75306b282b1ab8cd0b63b))
* Re-arrange nav (GITBOOK-398) ([54a87eb](https://github.com/jlopatec/phoenix/commit/54a87eb598c1eb01576a4e5baf3ddfa1f67322e8))
* re-organize integrations ([f90fb50](https://github.com/jlopatec/phoenix/commit/f90fb50bb836e9c56ff00cd98cf6024cafe17c5c))
* Ref Link Evals (GITBOOK-358) ([8488b4a](https://github.com/jlopatec/phoenix/commit/8488b4a1c2f66f3b5bd999d8641e6fb6a31a6970))
* remove lxml dep note (GITBOOK-331) ([d3c57e0](https://github.com/jlopatec/phoenix/commit/d3c57e059eba285a71543b965247a6d2b279974c))
* Remove the word golden, simplify title (GITBOOK-395) ([a2233b2](https://github.com/jlopatec/phoenix/commit/a2233b26b648466f60565c09286f8650823cf97f))
* remove word instrospection ([1fd04ee](https://github.com/jlopatec/phoenix/commit/1fd04eeb127fee8bc9afa5062988556fd3ee08c5))
* reorder notebooks (GITBOOK-279) ([6e56b23](https://github.com/jlopatec/phoenix/commit/6e56b2324e3ce1f00cd05520247204186aeddc04))
* Results (GITBOOK-339) ([bd9cb3a](https://github.com/jlopatec/phoenix/commit/bd9cb3a0c83a7bd739f569f75c4b5bf2195d6772))
* Retrieval Chunks (GITBOOK-372) ([39976d3](https://github.com/jlopatec/phoenix/commit/39976d3f020bfaaf0929f3bc8cbedb65d5aae010))
* Revise image links (GITBOOK-346) ([f108b24](https://github.com/jlopatec/phoenix/commit/f108b2443ef9bc0fc9f08a77d03ce61a01a64cea))
* run_relevance_eval (GITBOOK-332) ([299a625](https://github.com/jlopatec/phoenix/commit/299a6250c076aaafe58ad19074e470cb9ced8187))
* simplify conceps (GITBOOK-384) ([c38f6c2](https://github.com/jlopatec/phoenix/commit/c38f6c2cc1b47dfadd881e2ad9923f3d559aff27))
* Simplify examples page (GITBOOK-400) ([6144158](https://github.com/jlopatec/phoenix/commit/614415882527801ea23fb05eec7e01e12a190807))
* Simplify LLM Traces (GITBOOK-299) ([5346bfb](https://github.com/jlopatec/phoenix/commit/5346bfbe89c71dfbbfb9b79ef27160e89acf1bce))
* simplify name of models section (GITBOOK-326) ([c552f7b](https://github.com/jlopatec/phoenix/commit/c552f7b19bfaf96f17505085d3b146f2f6ede96f))
* Simplify the overview (GITBOOK-307) ([f09bcd3](https://github.com/jlopatec/phoenix/commit/f09bcd3b19004753a189da1ce16671f553339da9))
* Spelling fixes on LLM Traces (GITBOOK-311) ([82e3b81](https://github.com/jlopatec/phoenix/commit/82e3b81c55ad42bc810dcba2b8b3f89e67dc9dad))
* switch to the instrumentor for langchain ([7b474c8](https://github.com/jlopatec/phoenix/commit/7b474c836f962fc8288a4c0911f46d775c0fd2fe))
* sync ([#1947](https://github.com/jlopatec/phoenix/issues/1947)) ([c72bbac](https://github.com/jlopatec/phoenix/commit/c72bbacfbe23aba03167e58077e7b51bc2cde2e2))
* sync for 1.3 ([#1833](https://github.com/jlopatec/phoenix/issues/1833)) ([4d01e83](https://github.com/jlopatec/phoenix/commit/4d01e83edb7995ef07d02573d59060be9dba0fc1))
* sync from docs to main 10/16/2023 ([#1631](https://github.com/jlopatec/phoenix/issues/1631)) ([54f430f](https://github.com/jlopatec/phoenix/commit/54f430fb951fb0ef13939c3dc01007a5a6bd0c5d))
* Trace Evaluations Section (GITBOOK-388) ([2ffa800](https://github.com/jlopatec/phoenix/commit/2ffa8002d3a3a58a6f2b5e387129cd04f127baca))
* Traces UI Documentation (GITBOOK-292) ([099f363](https://github.com/jlopatec/phoenix/commit/099f363efee736aca32d3c8653acb1c2bef36387))
* **traces:** autogen tracing tutorial ([#1945](https://github.com/jlopatec/phoenix/issues/1945)) ([0fd02ff](https://github.com/jlopatec/phoenix/commit/0fd02ff4548342123f349f3e0b0957eac73d676c))
* tracing notebook updates ([#2053](https://github.com/jlopatec/phoenix/issues/2053)) ([a1e5323](https://github.com/jlopatec/phoenix/commit/a1e53238e03e8ade931084633f29e55dc1f8dd08))
* **tracing:** update langchain notebooks to use the instrumentor ([#1495](https://github.com/jlopatec/phoenix/issues/1495)) ([c76792a](https://github.com/jlopatec/phoenix/commit/c76792a22c2d4e60f7c40389c48600b3ea81142c))
* typos (GITBOOK-321) ([9e2ab7c](https://github.com/jlopatec/phoenix/commit/9e2ab7c802177627d71fc4cd0376569ef39a156e))
* update default value of variable in run_relevance_eval (GITBOOK-368) ([d5bcaf8](https://github.com/jlopatec/phoenix/commit/d5bcaf8147475f329cb55223cb981eb38611423c))
* update description for notebook (GITBOOK-278) ([7d9a4d0](https://github.com/jlopatec/phoenix/commit/7d9a4d078b8c20fe183ac45f3bfe1090c1d10d05))
* update development.md to highlight testing and linting instructions ([#1450](https://github.com/jlopatec/phoenix/issues/1450)) ([df2ecf4](https://github.com/jlopatec/phoenix/commit/df2ecf446d0d481855f027886d2001ab6210dad4))
* update docs for `phoenix.experimental.evals.llm_classify` (GITBOOK-356) ([5c4eb17](https://github.com/jlopatec/phoenix/commit/5c4eb17562d2101e673c05acf592dd8078c5bfed))
* update how-to (GITBOOK-312) ([5bc8d6d](https://github.com/jlopatec/phoenix/commit/5bc8d6d4b704775d7edebd3ac44b8217ae873698))
* Update image for span kinds (GITBOOK-349) ([891ca67](https://github.com/jlopatec/phoenix/commit/891ca67d7633c79fa4c547591ccdfcb082ed8cf4))
* Update Phoenix Inferences Quickstart (GITBOOK-357) ([5aefb0d](https://github.com/jlopatec/phoenix/commit/5aefb0d5407ad57e44a2a54b8a9b4b97666792b8))
* update rag eval notebook ([#1950](https://github.com/jlopatec/phoenix/issues/1950)) ([d06b8b7](https://github.com/jlopatec/phoenix/commit/d06b8b7cc68b9e0bcb9728cafd68e10b36b4d83e))
* update rag evals docs ([#1954](https://github.com/jlopatec/phoenix/issues/1954)) ([aa6f36a](https://github.com/jlopatec/phoenix/commit/aa6f36a3a5329b64fcb8f94d146ec33f5255a6eb))
* update RAG terminology ([#1489](https://github.com/jlopatec/phoenix/issues/1489)) ([9bc0414](https://github.com/jlopatec/phoenix/commit/9bc0414c433d39cd445e4cd389e7339ac8a34129))
* Update README.md with file extensions ([641c86b](https://github.com/jlopatec/phoenix/commit/641c86b078915612d70633c8b202a747100b38e9))
* Update SECURITY.md ([#2029](https://github.com/jlopatec/phoenix/issues/2029)) ([363e891](https://github.com/jlopatec/phoenix/commit/363e8913a3a0f7dafca5dc6bba6bf0e9776c1158))
* Update Session API for traces (GITBOOK-291) ([cef3f69](https://github.com/jlopatec/phoenix/commit/cef3f69ba6a33d19c1dd692c1deb4d889b77f025))
* updated traces images (GITBOOK-301) ([115f73d](https://github.com/jlopatec/phoenix/commit/115f73d9290e474fad26d64b675427929e97c6c8))
* updates eval notebook links (GITBOOK-274) ([23a5f51](https://github.com/jlopatec/phoenix/commit/23a5f51cd59ab55fc7222ab47ae4323b1307d482))
* Use Case Docs Take 2 (GITBOOK-290) ([5c35e6e](https://github.com/jlopatec/phoenix/commit/5c35e6e1e2db6ebeb1fb1702eeb0809d0959a4c2))
* use one-click for llama_index ([#1527](https://github.com/jlopatec/phoenix/issues/1527)) ([be6ac5c](https://github.com/jlopatec/phoenix/commit/be6ac5cff630b94d22c3739a9cfbe957b2a95c24))
* Using phoenix with HuggingFace LLMs- getting started ([#1916](https://github.com/jlopatec/phoenix/issues/1916)) ([b446972](https://github.com/jlopatec/phoenix/commit/b44697268a1e9599a4a155b9f0e283bb1a9ad349))
* **vals:** API documentaiton ([cc9c8f3](https://github.com/jlopatec/phoenix/commit/cc9c8f39077befcfe25ffac0739c9d5182ddeef1))
* Vertex Model small fix (GITBOOK-338) ([5594848](https://github.com/jlopatec/phoenix/commit/5594848ef430d411225cae3c5aa84ee4489f762d))

## [2.5.0](https://github.com/Arize-ai/phoenix/compare/v2.4.1...v2.5.0) (2024-01-16)


### Features

* **app:** databricks notebook support ([#2086](https://github.com/Arize-ai/phoenix/issues/2086)) ([b517480](https://github.com/Arize-ai/phoenix/commit/b517480e18d5e5d4c10d237cf13aff7546aa6529))


### Bug Fixes

* Adjust evaluation templates and rails for Gemini compatibility ([#2075](https://github.com/Arize-ai/phoenix/issues/2075)) ([3a7bfd2](https://github.com/Arize-ai/phoenix/commit/3a7bfd2158b5c0ea995f4d1ec00a6f7b2c4a21e8))

## [2.4.1](https://github.com/Arize-ai/phoenix/compare/v2.4.0...v2.4.1) (2024-01-11)


### Bug Fixes

* **traces:** prevent missing key exception when extracting invocation parameters in llama-index ([#2076](https://github.com/Arize-ai/phoenix/issues/2076)) ([5cc9560](https://github.com/Arize-ai/phoenix/commit/5cc956057f01613ebd08b0247d773c74b24a5aa3))

## [2.4.0](https://github.com/Arize-ai/phoenix/compare/v2.3.0...v2.4.0) (2024-01-10)


### Features

* add persistence for span evaluations ([#2021](https://github.com/Arize-ai/phoenix/issues/2021)) ([589d482](https://github.com/Arize-ai/phoenix/commit/589d482419b1119edfae24f837fd17d57612835f))
* **ui:** add filter condition snippets ([#2049](https://github.com/Arize-ai/phoenix/issues/2049)) ([567fa54](https://github.com/Arize-ai/phoenix/commit/567fa5499a4584e03bf1695caea3179778d27889))


### Bug Fixes

* Handle missing vertex candidates ([#2055](https://github.com/Arize-ai/phoenix/issues/2055)) ([1d0475a](https://github.com/Arize-ai/phoenix/commit/1d0475afa4974eb9590c70334961996f99e6d856))
* OpenAI clients are not cleaned up after calls to `llm_classify` ([#2068](https://github.com/Arize-ai/phoenix/issues/2068)) ([3233d56](https://github.com/Arize-ai/phoenix/commit/3233d561783556ef8799e65fe1335da52d161a12))
* **traces:** remove nan from log_evaluations ([#2056](https://github.com/Arize-ai/phoenix/issues/2056)) ([df9ed5c](https://github.com/Arize-ai/phoenix/commit/df9ed5cfd68da1e42d52b01aa9f85d68698c3b64))


### Documentation

* tracing notebook updates ([#2053](https://github.com/Arize-ai/phoenix/issues/2053)) ([a1e5323](https://github.com/Arize-ai/phoenix/commit/a1e53238e03e8ade931084633f29e55dc1f8dd08))

## [2.3.0](https://github.com/Arize-ai/phoenix/compare/v2.2.1...v2.3.0) (2024-01-08)


### Features

* evaluator enhancements ([#2045](https://github.com/Arize-ai/phoenix/issues/2045)) ([1cc9c0a](https://github.com/Arize-ai/phoenix/commit/1cc9c0a7a32c5896ff8a83fe25defaeb01fd9bf2))


### Bug Fixes

* Remove LiteLLM model support check ([#2046](https://github.com/Arize-ai/phoenix/issues/2046)) ([45d3fe6](https://github.com/Arize-ai/phoenix/commit/45d3fe6e1a2aaf931eb783aaca68525579215386))


### Documentation

* Add demo link, examples getting started (GITBOOK-396) ([e987315](https://github.com/Arize-ai/phoenix/commit/e987315ab35550350921257aa3e38f5cb6796f3f))
* Add Evaluating Traces Section (GITBOOK-386) ([7d72029](https://github.com/Arize-ai/phoenix/commit/7d720293cc30e0021596c427081e3b6bef4b7f5b))
* Add evaluations section for results (GITBOOK-387) ([2e74be0](https://github.com/Arize-ai/phoenix/commit/2e74be0a70a63ddfd6e42d84042f51fc9894f1cb))
* Add final thoughts to evaluation (GITBOOK-405) ([20eab16](https://github.com/Arize-ai/phoenix/commit/20eab16c2dc99b3c6defd497120ee2472eb97ff1))
* add import statement (GITBOOK-408) ([23247d7](https://github.com/Arize-ai/phoenix/commit/23247d755bad33b618e683687dd297e95274939c))
* add link (GITBOOK-403) ([0be280a](https://github.com/Arize-ai/phoenix/commit/0be280a974e7b8e88904c35285b149cde2369804))
* eval concepts typo (GITBOOK-394) ([7c80d4b](https://github.com/Arize-ai/phoenix/commit/7c80d4b3a151ee3a9b1937d04b7362ab2dba3697))
* eval concepts typos (GITBOOK-393) ([62bc99f](https://github.com/Arize-ai/phoenix/commit/62bc99f9e4d729e3bd911f346ce58bf44141c30f))
* evaluation concepts typo fix (GITBOOK-390) ([2cbc1dc](https://github.com/Arize-ai/phoenix/commit/2cbc1dcce6ac781d71fbf3107e6af74121be018e))
* Extract Data from Spans (GITBOOK-383) ([440f530](https://github.com/Arize-ai/phoenix/commit/440f530e35c13c7be1f69a05b0819afec094c5ed))
* fix broken section link (GITBOOK-409) ([fee537b](https://github.com/Arize-ai/phoenix/commit/fee537b9ede6a8d7899744b834a46066f7f8837d))
* fix typos (GITBOOK-391) ([c8f5a55](https://github.com/Arize-ai/phoenix/commit/c8f5a55d11669b3837c32cdbc10f99388f63276e))
* fix typos (GITBOOK-402) ([3cd973d](https://github.com/Arize-ai/phoenix/commit/3cd973d8b0348bb580455f093cc36f58deecf1b9))
* fix typos (GITBOOK-406) ([eaa9bea](https://github.com/Arize-ai/phoenix/commit/eaa9beac66cd72ad99ab5a8bc2aec95e2a4a0e7f))
* fix typos (GITBOOK-407) ([cad4820](https://github.com/Arize-ai/phoenix/commit/cad4820c085c7799adcf8d72b4209440b37fcbb9))
* Initial draft of evaluation core concept (GITBOOK-385) ([67369cf](https://github.com/Arize-ai/phoenix/commit/67369cf34e5eaddf30e0e3093f163483643f14bb))
* Log Evaluations (GITBOOK-389) ([369d79d](https://github.com/Arize-ai/phoenix/commit/369d79d479b4b094dbeeef8715b6127babb1a48c))
* No subject (GITBOOK-399) ([94df884](https://github.com/Arize-ai/phoenix/commit/94df884aa6f7ee98610026d827d48b57c0ba032b))
* Re-arrange nav (GITBOOK-398) ([54a87eb](https://github.com/Arize-ai/phoenix/commit/54a87eb598c1eb01576a4e5baf3ddfa1f67322e8))
* Remove the word golden, simplify title (GITBOOK-395) ([a2233b2](https://github.com/Arize-ai/phoenix/commit/a2233b26b648466f60565c09286f8650823cf97f))
* simplify conceps (GITBOOK-384) ([c38f6c2](https://github.com/Arize-ai/phoenix/commit/c38f6c2cc1b47dfadd881e2ad9923f3d559aff27))
* Simplify examples page (GITBOOK-400) ([6144158](https://github.com/Arize-ai/phoenix/commit/614415882527801ea23fb05eec7e01e12a190807))
* Trace Evaluations Section (GITBOOK-388) ([2ffa800](https://github.com/Arize-ai/phoenix/commit/2ffa8002d3a3a58a6f2b5e387129cd04f127baca))
* Update SECURITY.md ([#2029](https://github.com/Arize-ai/phoenix/issues/2029)) ([363e891](https://github.com/Arize-ai/phoenix/commit/363e8913a3a0f7dafca5dc6bba6bf0e9776c1158))

## [2.2.1](https://github.com/Arize-ai/phoenix/compare/v2.2.0...v2.2.1) (2023-12-28)


### Bug Fixes

* Do not retry if eval was successful when using SyncExecutor ([#2016](https://github.com/Arize-ai/phoenix/issues/2016)) ([a869190](https://github.com/Arize-ai/phoenix/commit/a8691905e16fe1c8c6483d837399e8c499ce71cf))
* ensure float values are properly encoded by otel tracer ([#2024](https://github.com/Arize-ai/phoenix/issues/2024)) ([b12a894](https://github.com/Arize-ai/phoenix/commit/b12a89496468063e2cd6a99e1ea08d49af5c6ba1))
* ensure llamaindex spans are correctly encoded ([#2023](https://github.com/Arize-ai/phoenix/issues/2023)) ([3ca6262](https://github.com/Arize-ai/phoenix/commit/3ca6262ad38c13b63dc2cde84e1a97a76ba1c323))
* Use separate versioning file ([#2020](https://github.com/Arize-ai/phoenix/issues/2020)) ([f38eedf](https://github.com/Arize-ai/phoenix/commit/f38eedfbd57da6e9512f9aa9fcc5c97232b2bb6e))

## [2.2.0](https://github.com/Arize-ai/phoenix/compare/v2.1.0...v2.2.0) (2023-12-22)


### Features

* Add support for Google's Gemini models via Vertex python sdk ([#2008](https://github.com/Arize-ai/phoenix/issues/2008)) ([caf826c](https://github.com/Arize-ai/phoenix/commit/caf826c8ced9f8840d3150f7d466bdc2295d2054))
* Support first-party Anthropic python SDK ([#2004](https://github.com/Arize-ai/phoenix/issues/2004)) ([a323283](https://github.com/Arize-ai/phoenix/commit/a323283021372bd3b7c95b435a8bef347e12028f))

## [2.1.0](https://github.com/Arize-ai/phoenix/compare/v2.0.0...v2.1.0) (2023-12-21)


### Features

* instantiate evaluators by criteria ([#1983](https://github.com/Arize-ai/phoenix/issues/1983)) ([9c72616](https://github.com/Arize-ai/phoenix/commit/9c72616aee88116c0194937d136bf3a74817132b))
* support function calling for run_evals ([#1978](https://github.com/Arize-ai/phoenix/issues/1978)) ([8be325c](https://github.com/Arize-ai/phoenix/commit/8be325cef3d48ce98f8c27c29dc46a42516ab970))
* **traces:** add `v1/traces` HTTP endpoint to handle `ExportTraceServiceRequest` ([3c94dea](https://github.com/Arize-ai/phoenix/commit/3c94dea6f5986c7b1658462d4700d21df56f72f6))
* **traces:** add `v1/traces` HTTP endpoint to handle `ExportTraceServiceRequest` ([#1968](https://github.com/Arize-ai/phoenix/issues/1968)) ([3c94dea](https://github.com/Arize-ai/phoenix/commit/3c94dea6f5986c7b1658462d4700d21df56f72f6))
* **traces:** add retrieval summary to header ([#2006](https://github.com/Arize-ai/phoenix/issues/2006)) ([8af0582](https://github.com/Arize-ai/phoenix/commit/8af0582c8c5cb19fe6bd84408f3d8ef26480c19e))
* **traces:** evaluation summary on the header ([#2000](https://github.com/Arize-ai/phoenix/issues/2000)) ([965beb0](https://github.com/Arize-ai/phoenix/commit/965beb00c223b405cd8c56c6a58d15beb391af78))


### Bug Fixes

* make alert icon for exceptions visible ([#2001](https://github.com/Arize-ai/phoenix/issues/2001)) ([e7a6567](https://github.com/Arize-ai/phoenix/commit/e7a6567f813149e4e7952f4b80183f39ea7cc15f))

## [2.0.0](https://github.com/Arize-ai/phoenix/compare/v1.9.0...v2.0.0) (2023-12-20)


### ⚠ BREAKING CHANGES

* Update `llm_classify` and `llm_generate` interfaces ([#1974](https://github.com/Arize-ai/phoenix/issues/1974))

### Features

* Add async submission to `llm_generate` ([#1965](https://github.com/Arize-ai/phoenix/issues/1965)) ([5999133](https://github.com/Arize-ai/phoenix/commit/59991331c93d4f994f00745124b73602d84cc95d))
* add support for explanations to run_evals ([#1975](https://github.com/Arize-ai/phoenix/issues/1975)) ([5143529](https://github.com/Arize-ai/phoenix/commit/51435297538c83542f7df6ba5860a50ef71307cf))
* evaluation column selectors ([#1932](https://github.com/Arize-ai/phoenix/issues/1932)) ([ed07809](https://github.com/Arize-ai/phoenix/commit/ed07809910e8b97387ef0729d843677e7e8c68a7))
* openai streaming tool calls ([#1936](https://github.com/Arize-ai/phoenix/issues/1936)) ([6dd14cf](https://github.com/Arize-ai/phoenix/commit/6dd14cf21acb12b599107619d0b7f4b0cdc55f4c))
* support running multiple evals at once ([#1742](https://github.com/Arize-ai/phoenix/issues/1742)) ([79d4473](https://github.com/Arize-ai/phoenix/commit/79d44739ba75b980967440a91850c9bed01ceb99))
* Update `llm_classify` and `llm_generate` interfaces ([#1974](https://github.com/Arize-ai/phoenix/issues/1974)) ([9fd35a1](https://github.com/Arize-ai/phoenix/commit/9fd35a11399c2d1aba1be40592cac0e31fff2d80))


### Bug Fixes

* Add lock failsafe ([#1956](https://github.com/Arize-ai/phoenix/issues/1956)) ([9ddbd9c](https://github.com/Arize-ai/phoenix/commit/9ddbd9caa4a66ae47c5447aa4dafa46cc9c5cfa5))
* llama-index extra ([#1958](https://github.com/Arize-ai/phoenix/issues/1958)) ([d9b68eb](https://github.com/Arize-ai/phoenix/commit/d9b68eb2e8f3422a491dc8a0981dd474d84d6260))
* LlamaIndex compatibility fix ([#1940](https://github.com/Arize-ai/phoenix/issues/1940)) ([052349d](https://github.com/Arize-ai/phoenix/commit/052349d594fa2d25b827f5318e0b438f33bf165c))
* Model stability enhancements ([#1939](https://github.com/Arize-ai/phoenix/issues/1939)) ([dca42e0](https://github.com/Arize-ai/phoenix/commit/dca42e026acc079f0523158d10e2149b01136600))
* **traces:** span summary root span filter ([#1981](https://github.com/Arize-ai/phoenix/issues/1981)) ([d286f07](https://github.com/Arize-ai/phoenix/commit/d286f077e8493c98a795ae83066e8c1c43a5291f))


### Documentation

* Add anyscale tutorial ([#1941](https://github.com/Arize-ai/phoenix/issues/1941)) ([e47c8d0](https://github.com/Arize-ai/phoenix/commit/e47c8d0be941f82ac13c5686fe75622cb59974ab))
* autogen link ([#1946](https://github.com/Arize-ai/phoenix/issues/1946)) ([c3fb4ce](https://github.com/Arize-ai/phoenix/commit/c3fb4ce80005c41201824114400abcdf007574c0))
* Clear anyscale tutorial outputs ([#1942](https://github.com/Arize-ai/phoenix/issues/1942)) ([63580a6](https://github.com/Arize-ai/phoenix/commit/63580a6bde204247953a920e8ca13209bd1d7f0b))
* RAG Evaluation (GITBOOK-378) ([429f537](https://github.com/Arize-ai/phoenix/commit/429f537ee034639dc66c4c80b455b22c65a2e7bd))
* sync ([#1947](https://github.com/Arize-ai/phoenix/issues/1947)) ([c72bbac](https://github.com/Arize-ai/phoenix/commit/c72bbacfbe23aba03167e58077e7b51bc2cde2e2))
* **traces:** autogen tracing tutorial ([#1945](https://github.com/Arize-ai/phoenix/issues/1945)) ([0fd02ff](https://github.com/Arize-ai/phoenix/commit/0fd02ff4548342123f349f3e0b0957eac73d676c))
* update rag eval notebook ([#1950](https://github.com/Arize-ai/phoenix/issues/1950)) ([d06b8b7](https://github.com/Arize-ai/phoenix/commit/d06b8b7cc68b9e0bcb9728cafd68e10b36b4d83e))
* update rag evals docs ([#1954](https://github.com/Arize-ai/phoenix/issues/1954)) ([aa6f36a](https://github.com/Arize-ai/phoenix/commit/aa6f36a3a5329b64fcb8f94d146ec33f5255a6eb))
* Using phoenix with HuggingFace LLMs- getting started ([#1916](https://github.com/Arize-ai/phoenix/issues/1916)) ([b446972](https://github.com/Arize-ai/phoenix/commit/b44697268a1e9599a4a155b9f0e283bb1a9ad349))

## [1.9.0](https://github.com/Arize-ai/phoenix/compare/v1.8.0...v1.9.0) (2023-12-11)


### Features

* Add retries to Bedrock ([#1927](https://github.com/Arize-ai/phoenix/issues/1927)) ([2728c3e](https://github.com/Arize-ai/phoenix/commit/2728c3e75927ca34e05c83336b3a8e9f5476466e))


### Documentation

* Add LLM Tracing+Evals notebook with keyless example ([#1928](https://github.com/Arize-ai/phoenix/issues/1928)) ([4c4aac6](https://github.com/Arize-ai/phoenix/commit/4c4aac6425af851b68f52d537813a8a1293a2a4b))

## [1.8.0](https://github.com/Arize-ai/phoenix/compare/v1.7.0...v1.8.0) (2023-12-10)


### Features

* **embeddings:** audio support ([#1920](https://github.com/Arize-ai/phoenix/issues/1920)) ([61cc550](https://github.com/Arize-ai/phoenix/commit/61cc55074c7381746886131c19e06d92a33f8489))
* openai streaming function call message support ([#1914](https://github.com/Arize-ai/phoenix/issues/1914)) ([25279ca](https://github.com/Arize-ai/phoenix/commit/25279ca563a81e438b7bbc3fd897d13ecca67b60))

## [1.7.0](https://github.com/Arize-ai/phoenix/compare/v1.6.0...v1.7.0) (2023-12-09)


### Features

* Instrument LlamaIndex streaming responses ([#1901](https://github.com/Arize-ai/phoenix/issues/1901)) ([f46396e](https://github.com/Arize-ai/phoenix/commit/f46396e04976475220092249a4e83f252d319630))
* openai async streaming instrumentation ([#1900](https://github.com/Arize-ai/phoenix/issues/1900)) ([06d643b](https://github.com/Arize-ai/phoenix/commit/06d643b7c7255b79c7a7e4ea587b4e445122ac37))
* **traces:** query spans into dataframes ([#1910](https://github.com/Arize-ai/phoenix/issues/1910)) ([6b51435](https://github.com/Arize-ai/phoenix/commit/6b5143535cf4ad3d0149fb68234043d47debaa15))


### Bug Fixes

* **traces:** span evaluations missing from the header ([#1908](https://github.com/Arize-ai/phoenix/issues/1908)) ([5ace81e](https://github.com/Arize-ai/phoenix/commit/5ace81e1a99afc72b6baf6464f1c21eea05eecdd))

## [1.6.0](https://github.com/Arize-ai/phoenix/compare/v1.5.1...v1.6.0) (2023-12-08)


### Features

* openai streaming spans show up in the ui ([#1888](https://github.com/Arize-ai/phoenix/issues/1888)) ([ffa1d41](https://github.com/Arize-ai/phoenix/commit/ffa1d41e633b6fee4978a9b705fa10bf4b5fe137))
* support instrumentation for openai synchronous streaming ([#1879](https://github.com/Arize-ai/phoenix/issues/1879)) ([b6e8c73](https://github.com/Arize-ai/phoenix/commit/b6e8c732926ea112775e9541173a9bdb29482d8d))
* **traces:** display document retrieval metrics on trace details ([#1902](https://github.com/Arize-ai/phoenix/issues/1902)) ([0c35229](https://github.com/Arize-ai/phoenix/commit/0c352297b3cef838651e69a05ae5357cbdbd61a5))
* **traces:** filterable span and document evaluation summaries ([#1880](https://github.com/Arize-ai/phoenix/issues/1880)) ([f90919c](https://github.com/Arize-ai/phoenix/commit/f90919c6162ce6bba3b12c5bba92b31f31128739))
* **traces:** graphql query for document evaluation summary ([#1874](https://github.com/Arize-ai/phoenix/issues/1874)) ([8a6a063](https://github.com/Arize-ai/phoenix/commit/8a6a06326e42f58018e030e0d854847d6fe6f10b))


### Documentation

* llm ops overview notebook ([#1882](https://github.com/Arize-ai/phoenix/issues/1882)) ([5d15c3c](https://github.com/Arize-ai/phoenix/commit/5d15c3c665583848624882e6d67979148673fca6))

## [1.5.1](https://github.com/Arize-ai/phoenix/compare/v1.5.0...v1.5.1) (2023-12-06)


### Bug Fixes

* Improve rate limiter behavior ([#1855](https://github.com/Arize-ai/phoenix/issues/1855)) ([2530569](https://github.com/Arize-ai/phoenix/commit/25305699c639d4c556c413e27a4c13378a548a77))

## [1.5.0](https://github.com/Arize-ai/phoenix/compare/v1.4.0...v1.5.0) (2023-12-06)


### Features

* **evals:** Human vs AI Evals ([#1850](https://github.com/Arize-ai/phoenix/issues/1850)) ([e96bd27](https://github.com/Arize-ai/phoenix/commit/e96bd27ed626a23187a92ddb34720a07ee689ad1))
* semantic conventions for `tool_calls` array in OpenAI ChatCompletion messages ([#1837](https://github.com/Arize-ai/phoenix/issues/1837)) ([c079f00](https://github.com/Arize-ai/phoenix/commit/c079f00fd731e281671bace9ef4b68d4cbdcc584))
* support asynchronous chat completions for openai instrumentation ([#1849](https://github.com/Arize-ai/phoenix/issues/1849)) ([f066e10](https://github.com/Arize-ai/phoenix/commit/f066e108c07c95502ba77b11fcb37fe3e5e5ed72))
* **traces:** document retrieval metrics based on document evaluation scores ([#1826](https://github.com/Arize-ai/phoenix/issues/1826)) ([3dfb7bd](https://github.com/Arize-ai/phoenix/commit/3dfb7bdfba3eb61e57dba503efcc761511479a90))
* **traces:** document retrieval metrics on trace / span tables ([#1873](https://github.com/Arize-ai/phoenix/issues/1873)) ([733d233](https://github.com/Arize-ai/phoenix/commit/733d2339ec3ffabc9ac83454fb6540adfefe1526))
* **traces:** evaluation annotations on traces for associating spans with eval metrics ([#1693](https://github.com/Arize-ai/phoenix/issues/1693)) ([a218a65](https://github.com/Arize-ai/phoenix/commit/a218a650cefa8925ed7b6627c121454bfc94ec0d))
* **traces:** server-side span filter by evaluation result values ([#1858](https://github.com/Arize-ai/phoenix/issues/1858)) ([6b05f96](https://github.com/Arize-ai/phoenix/commit/6b05f96fa7fc328414daf3caaed7d807e018763a))
* **traces:** span evaluation summary (aggregation metrics of scores and labels) ([#1846](https://github.com/Arize-ai/phoenix/issues/1846)) ([5c5c3d6](https://github.com/Arize-ai/phoenix/commit/5c5c3d69021fa21ce73a0d297107d5bf14fe4c98))


### Bug Fixes

* allow streaming response to be iterated by user ([#1862](https://github.com/Arize-ai/phoenix/issues/1862)) ([76a2443](https://github.com/Arize-ai/phoenix/commit/76a24436d7f2d1cb56ac77fb8486b4296f65a615))
* trace dataset to disc ([#1798](https://github.com/Arize-ai/phoenix/issues/1798)) ([278d344](https://github.com/Arize-ai/phoenix/commit/278d344434d43d5d05cc66abfcb9646b0ac2fb6d))


### Documentation

* RAG evaluation notebook using traces ([#1857](https://github.com/Arize-ai/phoenix/issues/1857)) ([4b67805](https://github.com/Arize-ai/phoenix/commit/4b67805931b059635997326de596d46a0bad1b76))
* Retrieval Chunks (GITBOOK-372) ([39976d3](https://github.com/Arize-ai/phoenix/commit/39976d3f020bfaaf0929f3bc8cbedb65d5aae010))

## [1.4.0](https://github.com/Arize-ai/phoenix/compare/v1.3.0...v1.4.0) (2023-11-30)


### Features

* propagate error status codes to parent spans for improved visibility into trace exceptions ([#1824](https://github.com/Arize-ai/phoenix/issues/1824)) ([1a234e9](https://github.com/Arize-ai/phoenix/commit/1a234e902d5882f19ab3c497e788bb2c4e2ff227))

## [1.3.0](https://github.com/Arize-ai/phoenix/compare/v1.2.1...v1.3.0) (2023-11-30)


### Features

* Add OpenAI Rate limiting ([#1805](https://github.com/Arize-ai/phoenix/issues/1805)) ([115e044](https://github.com/Arize-ai/phoenix/commit/115e04478f7192bdb4aa7b7a1cd0a5bd950fb03c))
* **evals:** show span evaluations in trace details slideout ([#1810](https://github.com/Arize-ai/phoenix/issues/1810)) ([4f0e4dc](https://github.com/Arize-ai/phoenix/commit/4f0e4dce35b779f2167581f281a0c70d61597f1d))
* evaluation ingestion (no user-facing feature is added) ([#1764](https://github.com/Arize-ai/phoenix/issues/1764)) ([7c4039b](https://github.com/Arize-ai/phoenix/commit/7c4039b3d9a04a73b312a09ceeb95a73de9610ef))
* feature flags context ([#1802](https://github.com/Arize-ai/phoenix/issues/1802)) ([a2732cd](https://github.com/Arize-ai/phoenix/commit/a2732cd115dad011c3856819fd2abf2a80ca2154))
* Implement asynchronous submission for OpenAI evals ([#1754](https://github.com/Arize-ai/phoenix/issues/1754)) ([30c011d](https://github.com/Arize-ai/phoenix/commit/30c011de471b68bc1a912780eff03ae0567d803e))
* reference link correctness evaluation prompt template ([#1771](https://github.com/Arize-ai/phoenix/issues/1771)) ([bf731df](https://github.com/Arize-ai/phoenix/commit/bf731df32d0f908b91a9f3ffb5ed6dcf6e00ff13))
* **traces:** configurable endpoint for the exporter ([#1795](https://github.com/Arize-ai/phoenix/issues/1795)) ([8515763](https://github.com/Arize-ai/phoenix/commit/851576385f548d8515e745bb39392fcf1b070e93))
* **traces:** display document evaluations alongside the document ([#1823](https://github.com/Arize-ai/phoenix/issues/1823)) ([2ca3613](https://github.com/Arize-ai/phoenix/commit/2ca361348ad112c6320c17b40d36bfadb0c2c66f))
* **traces:** server-side sort of spans by evaluation result (score or label) ([#1812](https://github.com/Arize-ai/phoenix/issues/1812)) ([d139693](https://github.com/Arize-ai/phoenix/commit/d1396931ab5b7b59c2777bb607afcf053134f6b7))
* **traces:** show all evaluations in the table" ([#1819](https://github.com/Arize-ai/phoenix/issues/1819)) ([2b27333](https://github.com/Arize-ai/phoenix/commit/2b273336b3448bc8cab1f433e79fc9fd868ad073))
* **traces:** Trace page header with latency, status, and evaluations ([#1831](https://github.com/Arize-ai/phoenix/issues/1831)) ([1d88efd](https://github.com/Arize-ai/phoenix/commit/1d88efdb623f5239106fde098fe51e53358592e2))


### Bug Fixes

* enhance llama-index callback support for exception events ([#1814](https://github.com/Arize-ai/phoenix/issues/1814)) ([8db01df](https://github.com/Arize-ai/phoenix/commit/8db01df096b5955adb12a57101beb76c943d8649))
* pin llama-index temporarily ([#1806](https://github.com/Arize-ai/phoenix/issues/1806)) ([d6aa76e](https://github.com/Arize-ai/phoenix/commit/d6aa76e2707528c367ea9ec5accc503871f97644))
* remove sklearn metrics not available in sagemaker ([#1791](https://github.com/Arize-ai/phoenix/issues/1791)) ([20ab6e5](https://github.com/Arize-ai/phoenix/commit/20ab6e551eb4de16df65d31d10b8efe34814c866))
* **traces:** convert (non-list) iterables to lists during protobuf construction due to potential presence of ndarray when reading from parquet files ([#1801](https://github.com/Arize-ai/phoenix/issues/1801)) ([ca72747](https://github.com/Arize-ai/phoenix/commit/ca72747991bf60881d76f95e88c656b1cecff2df))
* **traces:** make column selector sync'd between tabs ([#1816](https://github.com/Arize-ai/phoenix/issues/1816)) ([125431a](https://github.com/Arize-ai/phoenix/commit/125431a15cb9eaf07db406a936bd38c42f8665d8))


### Documentation

* Environment documentation (GITBOOK-370) ([dbbb0a7](https://github.com/Arize-ai/phoenix/commit/dbbb0a7cf86200d71327a2d29e889f31c1a0f149))
* Explanations (GITBOOK-371) ([5f33da3](https://github.com/Arize-ai/phoenix/commit/5f33da313625e5231619a447a8fbe0d173d638b0))
* No subject (GITBOOK-369) ([656b5c0](https://github.com/Arize-ai/phoenix/commit/656b5c0b9164517f78488b15dec14517590b4d5e))
* sync for 1.3 ([#1833](https://github.com/Arize-ai/phoenix/issues/1833)) ([4d01e83](https://github.com/Arize-ai/phoenix/commit/4d01e83edb7995ef07d02573d59060be9dba0fc1))
* update default value of variable in run_relevance_eval (GITBOOK-368) ([d5bcaf8](https://github.com/Arize-ai/phoenix/commit/d5bcaf8147475f329cb55223cb981eb38611423c))

## [1.2.1](https://github.com/Arize-ai/phoenix/compare/v1.2.0...v1.2.1) (2023-11-18)


### Bug Fixes

* make the app launchable when nest_asyncio is applied ([#1783](https://github.com/Arize-ai/phoenix/issues/1783)) ([f9d5085](https://github.com/Arize-ai/phoenix/commit/f9d508510c739007243ca200560268d53e6cb543))
* restore process session ([#1781](https://github.com/Arize-ai/phoenix/issues/1781)) ([34a32c3](https://github.com/Arize-ai/phoenix/commit/34a32c3e8567672bd1ac0979923566c39adecfcf))

## [1.2.0](https://github.com/Arize-ai/phoenix/compare/v1.1.1...v1.2.0) (2023-11-17)


### Features

* Add dockerfile ([#1761](https://github.com/Arize-ai/phoenix/issues/1761)) ([4fa8929](https://github.com/Arize-ai/phoenix/commit/4fa8929f4103e9961a8df0eb059b8df149ed648f))
* **evals:** return partial results when llm function is interrupted ([#1755](https://github.com/Arize-ai/phoenix/issues/1755)) ([1fb0849](https://github.com/Arize-ai/phoenix/commit/1fb0849a4e5f39c6afc90a1417300747a0bf4bf6))
* LiteLLM model support for evals ([#1675](https://github.com/Arize-ai/phoenix/issues/1675)) ([5f2a999](https://github.com/Arize-ai/phoenix/commit/5f2a9991059e060423853567a20789eba832f65a))
* sagemaker nobebook support ([#1772](https://github.com/Arize-ai/phoenix/issues/1772)) ([2c0ffbc](https://github.com/Arize-ai/phoenix/commit/2c0ffbc1479ae0255b72bc2d31d5f3204fd8e32c))


### Bug Fixes

* unpin llama-index version in tutorial notebooks ([#1766](https://github.com/Arize-ai/phoenix/issues/1766)) ([5ff74e3](https://github.com/Arize-ai/phoenix/commit/5ff74e3895f1b0c5642bd0897dd65e6f2913a7bd))


### Documentation

* add instructions for docker build ([#1770](https://github.com/Arize-ai/phoenix/issues/1770)) ([45eb5f2](https://github.com/Arize-ai/phoenix/commit/45eb5f244997d0ff0e991879c297b564e46c9a18))

## [1.1.1](https://github.com/Arize-ai/phoenix/compare/v1.1.0...v1.1.1) (2023-11-16)


### Bug Fixes

* update tracer for llama-index 0.9.0 ([#1750](https://github.com/Arize-ai/phoenix/issues/1750)) ([48d0996](https://github.com/Arize-ai/phoenix/commit/48d09960855d59419edfd10925aaa895fd370a0d))

## [1.1.0](https://github.com/Arize-ai/phoenix/compare/v1.0.0...v1.1.0) (2023-11-14)


### Features

* Evals with explanations ([#1699](https://github.com/Arize-ai/phoenix/issues/1699)) ([2db8141](https://github.com/Arize-ai/phoenix/commit/2db814102ea27f441e201740cc75ace79c82837c))
* **evals:** add an output_parser to llm_generate ([#1736](https://github.com/Arize-ai/phoenix/issues/1736)) ([6408dda](https://github.com/Arize-ai/phoenix/commit/6408dda7d33bfe84d929ddaacd01cb3269e0f63a))


### Documentation

* **evals:** document llm_generate with output parser ([#1741](https://github.com/Arize-ai/phoenix/issues/1741)) ([1e70ec3](https://github.com/Arize-ai/phoenix/commit/1e70ec3ff6158ffada40726419ae436ba6c7948d))

## [1.0.0](https://github.com/Arize-ai/phoenix/compare/v0.1.1...v1.0.0) (2023-11-10)


### ⚠ BREAKING CHANGES

* **models:** openAI 1.0 ([#1716](https://github.com/Arize-ai/phoenix/issues/1716))

### Features

* **models:** openAI 1.0 ([#1716](https://github.com/Arize-ai/phoenix/issues/1716)) ([2564521](https://github.com/Arize-ai/phoenix/commit/2564521e1ce00aaabaf592ce3735a656e1c6f1b8))

## [0.1.1](https://github.com/Arize-ai/phoenix/compare/v0.1.0...v0.1.1) (2023-11-09)


### Bug Fixes

* **traces:** handle `AIMessageChunk` in langchain tracer by matching prefix in name ([#1724](https://github.com/Arize-ai/phoenix/issues/1724)) ([8654c0a](https://github.com/Arize-ai/phoenix/commit/8654c0a0c9e088284b4ed0bbbae4f571ae11b1e7))

## [0.1.0](https://github.com/Arize-ai/phoenix/compare/v0.0.51...v0.1.0) (2023-11-08)


### Features

* add long-context evaluators, including map reduce and refine patterns ([#1710](https://github.com/Arize-ai/phoenix/issues/1710)) ([0c3b105](https://github.com/Arize-ai/phoenix/commit/0c3b1053f95b88e234ed3ccfffa50b27f48dc359))
* **traces:** span table column visibility controls ([#1687](https://github.com/Arize-ai/phoenix/issues/1687)) ([559852f](https://github.com/Arize-ai/phoenix/commit/559852f12d976df691c24f3e89bf23a7650d148c))


### Bug Fixes

* add bedrock import ([#1695](https://github.com/Arize-ai/phoenix/issues/1695)) ([dc7f3ef](https://github.com/Arize-ai/phoenix/commit/dc7f3ef6fa7c184e46ef92f1c035a29345e0b12e))
* pin openai version below 1.0.0 ([#1714](https://github.com/Arize-ai/phoenix/issues/1714)) ([d21e364](https://github.com/Arize-ai/phoenix/commit/d21e36459a42f01d04a373236fef2e603eea159d))
* **traces:** Keep traces visible behind the details slideover ([#1709](https://github.com/Arize-ai/phoenix/issues/1709)) ([1c8b8f1](https://github.com/Arize-ai/phoenix/commit/1c8b8f175e0cc8506490a8e0e36ad92d7e7ff69a))


### Documentation

* pin tutorials to openai&lt;1 ([#1718](https://github.com/Arize-ai/phoenix/issues/1718)) ([831c041](https://github.com/Arize-ai/phoenix/commit/831c041185e5f514fa1a13836758efecbf05e1dd))
