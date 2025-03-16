import React, { useEffect, useState }from "react";
import "./Home.css";
import doctorImage from "../../assets/Doctor.png"; // Replace with your actual image

const Home = () => {
  const words = ["Scroll Down To Fill The Form"];
  const [currentWord, setCurrentWord] = useState("");
  const [isDeleting, setIsDeleting] = useState(false);
  const [wordIndex, setWordIndex] = useState(0);
  const [typingSpeed, setTypingSpeed] = useState(200);

  useEffect(() => {
    const handleType = () => {
      const fullText = words[wordIndex];

      if (isDeleting) {
        setCurrentWord(fullText.substring(0, currentWord.length - 1));
        setTypingSpeed(100);
      } else {
        setCurrentWord(fullText.substring(0, currentWord.length + 1));
        setTypingSpeed(200);
      }

      if (!isDeleting && currentWord === fullText) {
        setTypingSpeed(2000);
        setIsDeleting(true);
      } else if (isDeleting && currentWord === "") {
        setIsDeleting(false);
        setWordIndex((prevIndex) => (prevIndex + 1) % words.length);
        setTypingSpeed(500);
      }
    };

    const timer = setTimeout(handleType, typingSpeed);
    return () => clearTimeout(timer);
  }, [currentWord, isDeleting, typingSpeed, wordIndex, words]);
  useEffect(() => {
    // Add animation class when component mounts
    document.querySelector('.home-container').style.opacity = '1';
  }, []);

  return (
    <div className="home-container">
      <div className="text-content">
        <h1>
        Predicting Success Of Pediatric <span className="highlight">Bone Marrow</span> Using AI
        </h1>
        <p>
          
        </p>
        <div className="stats">
        <div ><span className='stat-item'>{currentWord}</span></div>
        </div>
      </div>

      {/* Doctor Image Section */}
      <div className="image-container">
        <div className="circle-bg"></div>
        <img src={doctorImage} alt="Doctor" className="doctor-image" />
      </div>
    </div>
  );
};

export default Home;
