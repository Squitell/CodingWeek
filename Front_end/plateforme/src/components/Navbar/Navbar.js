import React, { useState } from 'react';
import './Navbar.css';
import logo from '../../assets/logo-ecc.png';

const Navbar = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const scrollToSection = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    setIsMobileMenuOpen(false); // Close mobile menu after clicking
  };

  return (
    <nav className='Navbar'>
      <img src={logo} alt='Logo' className='logo' />

      {/* Desktop Menu */}
      <ul className='desktopMenu'>
        <li className='desktopMenuLinkItem' onClick={() => scrollToSection('home')}>Home</li>
        <li className='desktopMenuLinkItem' onClick={() => scrollToSection('form')}>Form</li>
        <li className='desktopMenuLinkItem' onClick={() => scrollToSection('results')}>Results</li>
        <li className='desktopMenuLinkItem' onClick={() => scrollToSection('about')}>About Us</li>
      </ul>

      {/* Mobile Menu Button */}
      <div className="hamburger" onClick={toggleMobileMenu}>
        <div className={isMobileMenuOpen ? "bar open" : "bar"}></div>
        <div className={isMobileMenuOpen ? "bar open" : "bar"}></div>
        <div className={isMobileMenuOpen ? "bar open" : "bar"}></div>
      </div>

      {/* Mobile Menu */}
      <ul className={`mobileMenu ${isMobileMenuOpen ? 'open' : ''}`}>
        <li className='mobileMenuLinkItem' onClick={() => scrollToSection('home')}>Home</li>
        <li className='mobileMenuLinkItem' onClick={() => scrollToSection('form')}>Form</li>
        <li className='mobileMenuLinkItem' onClick={() => scrollToSection('about')}>About Us</li>
        <li className='mobileMenuLinkItem' onClick={() => scrollToSection('results')}>Results</li>
      </ul>
    </nav>
  );
};

export default Navbar;
