import Config

config :nx, default_backend: EXLA.Backend
config :logger, level: :debug

import_config "#{config_env()}.exs"
