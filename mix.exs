defmodule YOLO.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :yolo,
      version: @version,
      elixir: "~> 1.17",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps()
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
      {:image, "~> 0.54.4", only: [:dev, :test]},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false}
    ]
  end
end
