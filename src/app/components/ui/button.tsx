// Nel file button-test.js

import React from 'react';

export const ButtonsCard = ({
    children,
    className,
    onClick, // Aggiunto onClick come prop
}: {
    children?: React.ReactNode;
    className?: string;
    onClick?: () => void; // Definisci il tipo di prop onClick
}) => {
    return (
        <div className={`relative z-40 flex justify-center items-center h-full ${className}`}>
            {/* Aggiungi onClick all'elemento contenitore */}
            <div onClick={onClick}>{children}</div>
        </div>
    );
};


