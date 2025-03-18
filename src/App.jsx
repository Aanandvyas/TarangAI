import React from 'react'
import Navbar from './components/Navbar'
import Landing from './components/Landing'
import Nritya from './components/Nritya'
import PlayReel from './components/PlayReel'
import Images from './components/Images'
import Generate from './components/Generate'
import AboutPage from './components/AboutPage'


const App = () => {
  return (
    <div className='bg-zinc-200 w-screen h-screen overflow-y-auto overflow-x-hidden'>
      <div>
        <Navbar />
        <Landing/>
        <Nritya />
        <PlayReel/>
        <Images/>
        <Generate/>     
        <AboutPage/>   
      </div>
    </div>
  )
}

export default App