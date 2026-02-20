//
// Physical [row][col] == channel
//
// Generate the standardised layout.
// We'll want this to be mea layout aware when that is a thing.
//
// We should also migrate to a [col][row] format to match other
// areas of the system.
//

export const rowColChannelLayout = [];

for (let row = 0; row < 8; row++)
{
    const channels = [];
    for (let col = 0; col < 8; col++)
        channels.push(col * 8 + row);

    rowColChannelLayout.push(channels);
}