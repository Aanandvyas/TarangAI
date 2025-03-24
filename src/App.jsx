import React, { useEffect, useRef } from 'react';
import LocomotiveScroll from 'locomotive-scroll';

import Navbar from './components/Navbar';
import Landing from './components/Landing';
import Nritya from './components/Nritya';
import PlayReel from './components/PlayReel';
import Images from './components/Images';
import Generate from './components/Generate';
import AboutPage from './components/AboutPage';

const App = () => {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (!scrollRef.current) return;

    const scroll = new LocomotiveScroll({
      el: scrollRef.current,
      smooth: true,
      multiplier: 1.2,
      class: 'is-inview',
    });

    
    return () => {
      scroll.destroy(); // Clean up to avoid memory leaks
    };
  }, []);

  return (
    <div ref={scrollRef} data-scroll-container className="bg-zinc-200 w-screen overflow-x-hidden overflow-y-auto">
      <Navbar />
      <Landing />
      <Nritya />
      <PlayReel />
      <Images />
      <Generate />
      <AboutPage />
    </div>
  );
};

export default App;
