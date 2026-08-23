import { TrainingDemo } from "./m0/TrainingDemo";

export default function App() {
  return (
    <div className="app">
      <header>
        <h1>Grokking Nets</h1>
        <p className="tagline">
          An interactive course on neural networks: read a little, play with live
          visualizations, and implement the real thing in Python, right here in your
          browser.
        </p>
      </header>
      <main>
        <TrainingDemo />
      </main>
      <footer>
        <p>
          Adapted from Michael A. Nielsen,{" "}
          <a href="http://neuralnetworksanddeeplearning.com/">
            <em>Neural Networks and Deep Learning</em>
          </a>
          , Determination Press, 2015, licensed{" "}
          <a href="https://creativecommons.org/licenses/by-nc/3.0/">CC BY-NC 3.0</a>.
          This derivative work is non-commercial and inherits the same license.
        </p>
      </footer>
    </div>
  );
}
