import React, { useState, useEffect } from 'react';
import './App.css'; // Assuming this CSS file contains all the necessary styles
import io from 'socket.io-client';

const socket = io('http://localhost:4000');

const App = () => {
  const [backgroundClass, setBackgroundClass] = useState('');
  const [message, setMessage] = useState('');
  const [chat, setChat] = useState([]);
  const [currentQuestion, setCurrentQuestion] = useState("What's your favorite color?");
  const [answer, setAnswer] = useState('');

  useEffect(() => {
    const classes = ['bright-blue', 'bright-red', 'bright-green'];
    const randomClass = classes[Math.floor(Math.random() * classes.length)];
    setBackgroundClass(randomClass);

    socket.on('chat message', (msg) => {
      setChat((prevChat) => [...prevChat, msg]);
    });

    return () => {
      socket.off('chat message');
    };
  }, []); // Fixed to avoid unnecessary re-renders

  const sendMessage = (e) => {
    e.preventDefault();
    if (!message.trim()) return;
    socket.emit('chat message', message);
    setMessage('');
  };

  const handleAnswerSubmit = () => {
    if (!answer.trim()) return;

    const messageToSend = `Answer: ${answer}`;
    socket.emit('chat message', messageToSend);
    setChat((prevChat) => [...prevChat, messageToSend]);
    setAnswer(''); // Clear the answer input
  };

  return (
    <div className={backgroundClass}>
      <main className="container">
        <div className="question-section">
          <p>{currentQuestion}</p>
          <input
            type="text"
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            placeholder="Your answer..."
          />
          <button onClick={handleAnswerSubmit}>Submit Answer</button>
        </div>
        <div className="chat-section">
          <ul id="messages">
            {chat.map((msg, index) => (
              <li key={index}>{msg}</li>
            ))}
          </ul>
          <form onSubmit={sendMessage}>
            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Enter your 106B question here!"
            />
            <button type="submit">Send</button>
          </form>
        </div>
      </main>
    </div>
  );
};

export default App;
