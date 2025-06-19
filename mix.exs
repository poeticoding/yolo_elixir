defmodule YOLO.MixProject do
  use Mix.Project

  @source_url "https://github.com/poeticoding/yolo_elixir"
  @version "0.2.0"

  def project do
    [
      app: :yolo,
      version: @version,
      elixir: "~> 1.17",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      name: "YOLO",
      source_url: @source_url,

      # Hex
      description: description(),
      package: package(),

      # Docs
      docs: docs()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp elixirc_paths(:test), do: ~w(lib test/support)
  defp elixirc_paths(_), do: ~w(lib)

  defp deps do
    [
      # required
      {:ortex, "~> 0.1.10"},
      {:nx, "~> 0.9"},

      # code check and docs
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.35", only: :dev, runtime: false},

      # benchmarking
      {:benchee, "~> 1.3.0", only: :dev},

      # OPTIONAL
      {:exla, "~> 0.9", optional: true},
      {:yolo_fast_nms, "~> 0.2", optional: true},
      {:evision, "~> 0.2.0", optional: true},
      {:image, "~> 0.54.4", optional: true}
    ]
  end

  defp package do
    [
      name: "yolo",
      maintainers: ["Alvise Susmel"],
      files: ~w(mix.exs README.md LICENSE lib),
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end

  defp description do
    "A library for object detection and seamless YOLO model integration in Elixir"
  end

  defp docs do
    [
      main: "readme",
      source_ref: "v#{@version}",
      extras: extras(),
      assets: %{"guides/images" => "guides/images"},
      groups_for_extras: [
        "Examples": [
          "examples/ultralytics_yolo.livemd",
          "examples/yolox.livemd",
          "examples/webcam.livemd",
          "examples/yolo_oiv7.livemd"
        ],
        "Benchmarks": [
          "guides/benchmarks.md",
          "guides/benchmarks/macbook_air_m3.md"
        ]
      ]
    ]
  end

  defp extras do
    [
      "README.md",
      "guides/ultralytics_to_onnx.md",
      "guides/under_the_hood.md",

      "guides/benchmarks.md",
      "guides/benchmarks/macbook_air_m3.md",


      "examples/ultralytics_yolo.livemd",
      "examples/yolox.livemd",
      "examples/webcam.livemd",
      "examples/yolo_oiv7.livemd",

    ]
  end
end
