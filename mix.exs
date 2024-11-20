defmodule YOLO.MixProject do
  use Mix.Project

  @source_url "https://github.com/poeticoding/yolo_elixir"
  @version "0.1.0"

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
      package: package()
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
      {:ortex, "~> 0.1.9"},
      {:nx, "~> 0.9.1"},
      {:exla, "~> 0.9.1"},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.35", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      name: "yolo",
      maintainers: ["Alvise Susmel"],
      files: ~w(.formatter.exs mix.exs README.md lib priv),
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end

  defp description do
    "A library for object detection and seamless YOLO model integration in Elixir"
  end
end
