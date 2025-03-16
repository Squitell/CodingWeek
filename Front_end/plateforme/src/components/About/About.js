
import React from 'react';
import { motion } from 'framer-motion';
import './About.css';

const members = ['Ahmed Terraf', 'Samia Touile', 'Hamza Ouzahra', 'Mohammed Elhathot',"Emna Machouch"];



export default function About() {
  const angleStep = (2 * Math.PI) / members.length;

  return (
    <div className="about-page">
      <motion.div
        className="about-info"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
      >
        <h2>Predicting Success of Pediatric Bone Marrow Using AI</h2>
        <p className="group">TEAM 8</p>
      </motion.div>

      <div className="circle-container">
        {members.map((name, idx) => {
          const angle = idx * angleStep;
          const x = 200 * Math.cos(angle);
          const y = 200 * Math.sin(angle);

          return (
            <motion.div
              key={idx}
              className="circle-member"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1, x, y }}
              transition={{ duration: 0.7, delay: idx * 0.2 }}
            >
              <div className="avatar">{name[0]}</div>
              <div className="member-name">{name}</div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
