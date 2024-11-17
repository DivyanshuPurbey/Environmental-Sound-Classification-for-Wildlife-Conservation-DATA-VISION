import React from 'react';
import Navbar from './Navbar';
import AudioUpload from './Audioupload';
import './App.css'; // You can add global styles here

const App = () => {
  return (
    <div className="App">
      <Navbar />
      <AudioUpload />
    </div>
  );
};

export default App;