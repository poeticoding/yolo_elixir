import Config

config :nx,
  default_backend: {EXLA.Backend, client: :host},
  default_defn_options: [compiler: EXLA]

# CoreML acceleration for Mac
ortex_features =
  case :os.type() do
    {:win32, _} -> ["directml"]
    {:unix, :darwin} -> ["coreml"]
    {:unix, _} -> ["cuda", "tensorrt"]
  end

config :ortex, Ortex.Native, features: ortex_features
