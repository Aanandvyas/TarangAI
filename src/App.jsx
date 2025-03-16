import React from 'react'
import Navbar from './components/Navbar'
import Landing from './components/Landing'
import Nritya from './components/Nritya'


const App = () => {
  return (
    <div className='bg-zinc-900 w-screen h-screen'>
      <div>
        <Navbar />
        <Landing/>
        <Nritya />
      </div>
    </div>
  )
}

export default App