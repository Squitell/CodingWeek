import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar/Navbar';
import Home from './components/Home/Home';
import Form from './components/Form/Form';
import Results from './components/Results/Results';
import About from './components/About/About';
import Footer from './Footer/Footer';

const App = () => {
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.substring(1);
      const section = document.getElementById(hash);
      if (section) {
        section.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    };

    window.addEventListener("hashchange", handleHashChange);
    return () => {
      window.removeEventListener("hashchange", handleHashChange);
    };
  }, []);

  return (
    <div>
      <Navbar />
      <div id="home"><Home /></div>
      <div id="form"><Form setPrediction={setPrediction} /></div>
      <div id="results"><Results prediction={prediction} /></div>
      <div id="about"><About /></div>
      <div id="footer"><Footer /></div>
    </div>
  );
};

export default App;
