import React, { useState, useEffect }  from 'react';
import './App.css'; // Make sure to import the Pico CSS in this file or in index.js
const App = () => {
    const [backgroundClass, setBackgroundClass] = useState('');

  useEffect(() => {
    const classes = ['bright-blue', 'bright-red', 'bright-green']; // Ensure these classes are defined in your App.css
    const randomClass = classes[Math.floor(Math.random() * classes.length)];
    setBackgroundClass(randomClass);
  }, []);
  return (
    <div>
    <div className={backgroundClass}>
    </div>
      <nav className="container-fluid">
        <ul>
          <li><strong>Game Lobby</strong></li>
        </ul>
        <ul>
          <li><a href="#">Home</a></li>
          <li><a href="#">About</a></li>
          <li><a href="#" role="button">Contact</a></li>
        </ul>
      </nav>
      <main className="container">
        <div className="grid">
          <section>
            <hgroup>
              <h2>Welcome to the Quizzem!</h2>
              <h3>Get ready to play!</h3>
            </hgroup>
            <p>Your game will start shortly. Feel free to chat with other players while you wait!</p>
            <figure>
              <img src="https://source.unsplash.com/random/400x300" alt="Game waiting room" />
              <figcaption> </figcaption>
            </figure>
            <h3>Rules</h3>
            <p>Please read the game rules before starting.</p>
            <h3>Support</h3>
            <p>If you need help, contact us using the link above.</p>
          </section>
        </div>
      </main>
      <footer className="container">
        <small><a href="#">Privacy Policy</a> • <a href="#">Terms of Use</a></small>
      </footer>
    </div>)
}
export default App;
