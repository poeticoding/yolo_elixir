defmodule YOLO.MixProject do
  use Mix.Project

  def project do
    [
      app: :yolo,
      version: "0.1.0",
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
      {:image, "~> 0.54.4", only: [:dev, :test]}
    ]
  end
end
