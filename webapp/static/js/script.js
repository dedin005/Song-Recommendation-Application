// script.js
async function fetchSongs() {
    const url = document.getElementById('songUrl').value;
    const response = await fetch('/process_song', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ song_link: url }),
    });
    const songs = await response.json();

    const songsContainer = document.getElementById('songs');
    songsContainer.innerHTML = '';
    songs.forEach(song => {
        const songDiv = document.createElement('div');
        songDiv.className = 'max-w-md mx-auto bg-white rounded-xl shadow-md overflow-hidden md:max-w-2xl my-4 p-4';
        songDiv.innerHTML = `
            <div class="md:flex">
                <div class="p-4">
                    <p class="block mt-1 text-lg leading-tight font-medium text-black">${song.name} by ${song.artist}</p>
                    <audio controls src="${song.url_spotify_preview}" class="mt-2">
                        Your browser does not support the audio element.
                    </audio>
                </div>
            </div>`;
        songsContainer.appendChild(songDiv);
    });
}
