import React from 'react'
import Navbar from './components/Navbar'
import Landing from './components/Landing'

const App = () => {
  return (
    <div className='bg-zinc-900 w-screen h-screen'>
      <div>
        <Navbar />
        <Landing/>
      </div>
    </div>
  )
}

export default App